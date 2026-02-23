#include "tool-registry.h"
#include "tool-session-executor.h"

#include "log.h"
#include "server-common.h" // gen_tool_call_id(), random_string()

#include <chrono>
#include <stdexcept>

//
// Lifecycle
//

tool_registry::tool_registry() = default;

tool_registry::~tool_registry() {
    if (m_initialized.load()) {
        shutdown();
    }
}

bool tool_registry::initialize(const json & config) {
    if (config.contains("session_ttl_seconds")) {
        m_session_ttl = std::chrono::seconds(config["session_ttl_seconds"].get<int>());
    }

    // Start background cleanup thread
    m_cleanup_thread = std::thread([this]() {
        while (!m_shutdown_requested.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            if (!m_shutdown_requested.load()) {
                cleanup_expired_sessions();
            }
        }
    });

    m_initialized.store(true);
    LOG_INF("tool_registry: initialized (session_ttl=%llds)\n",
            (long long)m_session_ttl.count());
    return true;
}

void tool_registry::shutdown() {
    m_shutdown_requested.store(true);

    // Wake up cleanup thread
    if (m_cleanup_thread.joinable()) {
        m_cleanup_thread.join();
    }

    // Close all sessions
    {
        std::unique_lock slock(m_sessions_mutex);
        for (auto & kv : m_sessions) {
            // Notify session-aware executors
            std::shared_lock elock(m_executors_mutex);
            for (auto & ekv : m_executors) {
                auto * se = dynamic_cast<tool_session_executor *>(ekv.second.get());
                if (se) {
                    try {
                        se->on_session_end(kv.first);
                    } catch (const std::exception & ex) {
                        LOG_WRN("tool_registry: on_session_end threw: %s\n", ex.what());
                    } catch (...) {
                        LOG_WRN("tool_registry: on_session_end threw unknown exception\n");
                    }
                }
            }
        }
        m_sessions.clear();
    }

    // Shutdown all executors
    {
        std::unique_lock elock(m_executors_mutex);
        for (auto & kv : m_executors) {
            try {
                kv.second->shutdown();
            } catch (const std::exception & ex) {
                LOG_WRN("tool_registry: executor '%s' shutdown threw: %s\n",
                        kv.first.c_str(), ex.what());
            } catch (...) {
                LOG_WRN("tool_registry: executor '%s' shutdown threw unknown exception\n",
                        kv.first.c_str());
            }
        }
        m_executors.clear();
    }

    {
        std::unique_lock tlock(m_tools_mutex);
        m_tools.clear();
    }

    m_initialized.store(false);
    LOG_INF("tool_registry: shutdown complete\n");
}

//
// Executor management
//

void tool_registry::register_executor(tool_executor_ptr executor) {
    if (!executor) {
        throw std::invalid_argument("tool_registry: null executor");
    }

    auto name = executor->get_name();

    if (!executor->initialize(json::object())) {
        LOG_WRN("tool_registry: executor '%s' initialize() returned false\n", name.c_str());
    }

    auto provided = executor->list_provided_tools();

    {
        std::unique_lock lock(m_executors_mutex);
        m_executors[name] = std::move(executor);
    }

    for (const auto & def : provided) {
        register_tool(def, name);
    }

    LOG_INF("tool_registry: registered executor '%s' with %zu tools\n",
            name.c_str(), provided.size());
}

void tool_registry::unregister_executor(const std::string & name) {
    {
        std::unique_lock tlock(m_tools_mutex);
        for (auto it = m_tools.begin(); it != m_tools.end(); ) {
            if (it->second.executor_name == name) {
                it = m_tools.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::unique_lock elock(m_executors_mutex);
    auto it = m_executors.find(name);
    if (it != m_executors.end()) {
        try {
            it->second->shutdown();
        } catch (const std::exception & ex) {
            LOG_WRN("tool_registry: executor '%s' shutdown threw: %s\n",
                    name.c_str(), ex.what());
        } catch (...) {
            LOG_WRN("tool_registry: executor '%s' shutdown threw unknown exception\n",
                    name.c_str());
        }
        m_executors.erase(it);
    }
}

tool_executor * tool_registry::get_executor(const std::string & name) const {
    std::shared_lock lock(m_executors_mutex);
    auto it = m_executors.find(name);
    return it != m_executors.end() ? it->second.get() : nullptr;
}

//
// Tool management
//

void tool_registry::register_tool(const tool_definition & def, const std::string & executor_name) {
    tool_executor * exec_ptr = get_executor(executor_name);
    if (!exec_ptr) {
        throw std::runtime_error("tool_registry: executor '" + executor_name
                                 + "' not found when registering tool '" + def.name + "'");
    }

    std::unique_lock lock(m_tools_mutex);
    m_tools[def.name] = tool_registration{def, exec_ptr, executor_name};
    LOG_DBG("tool_registry: registered tool '%s' -> executor '%s'\n",
            def.name.c_str(), executor_name.c_str());
}

void tool_registry::unregister_tool(const std::string & name) {
    std::unique_lock lock(m_tools_mutex);
    m_tools.erase(name);
}

std::vector<tool_definition> tool_registry::list_tools() const {
    std::shared_lock lock(m_tools_mutex);
    std::vector<tool_definition> result;
    result.reserve(m_tools.size());
    for (const auto & kv : m_tools) {
        result.push_back(kv.second.definition);
    }
    return result;
}

std::optional<tool_definition> tool_registry::get_tool(const std::string & name) const {
    std::shared_lock lock(m_tools_mutex);
    auto it = m_tools.find(name);
    if (it == m_tools.end()) {
        return std::nullopt;
    }
    return it->second.definition;
}

bool tool_registry::has_tool(const std::string & name) const {
    std::shared_lock lock(m_tools_mutex);
    return m_tools.count(name) > 0;
}

//
// Stateless execution
//

tool_call_result tool_registry::execute_tool(const tool_call_request & request) {
    tool_call_result error;
    error.call_id = request.call_id;
    error.success = false;

    if (m_shutdown_requested.load()) {
        error.error_message = "Server is shutting down";
        error.error_code    = 503;
        return error;
    }

    tool_executor * exec_ptr = nullptr;
    {
        std::shared_lock lock(m_tools_mutex);
        auto it = m_tools.find(request.tool_name);
        if (it == m_tools.end()) {
            error.error_message = "Tool not found: " + request.tool_name;
            error.error_code    = 404;
            return error;
        }
        exec_ptr = it->second.executor;
    }

    if (!exec_ptr) {
        error.error_message = "Executor not available for tool: " + request.tool_name;
        error.error_code    = 500;
        return error;
    }

    if (!exec_ptr->is_healthy()) {
        error.error_message = "Executor unhealthy for tool: " + request.tool_name;
        error.error_code    = 503;
        error.is_retryable  = true;
        return error;
    }

    auto t0     = std::chrono::steady_clock::now();
    auto result = exec_ptr->execute(request);
    auto t1     = std::chrono::steady_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return result;
}

std::future<tool_call_result> tool_registry::execute_tool_async(const tool_call_request & request) {
    return std::async(std::launch::async, [this, request]() {
        return execute_tool(request);
    });
}

//
// Session management
//

std::string tool_registry::create_session(const json & context) {
    std::string sid = random_string();

    tool_session session;
    session.session_id = sid;
    session.context    = context;

    std::unique_lock lock(m_sessions_mutex);
    m_sessions[sid] = std::move(session);

    LOG_INF("tool_registry: created session '%s'\n", sid.c_str());
    return sid;
}

bool tool_registry::has_session(const std::string & session_id) const {
    std::shared_lock lock(m_sessions_mutex);
    return m_sessions.count(session_id) > 0;
}

bool tool_registry::close_session(const std::string & session_id) {
    std::unique_lock slock(m_sessions_mutex);
    auto it = m_sessions.find(session_id);
    if (it == m_sessions.end()) {
        return false;
    }
    m_sessions.erase(it);

    // Notify session-aware executors
    std::shared_lock elock(m_executors_mutex);
    for (auto & kv : m_executors) {
        auto * se = dynamic_cast<tool_session_executor *>(kv.second.get());
        if (se) {
            try {
                se->on_session_end(session_id);
            } catch (const std::exception & ex) {
                LOG_WRN("tool_registry: on_session_end threw: %s\n", ex.what());
            } catch (...) {
                LOG_WRN("tool_registry: on_session_end threw unknown exception\n");
            }
        }
    }

    LOG_INF("tool_registry: closed session '%s'\n", session_id.c_str());
    return true;
}

std::vector<std::string> tool_registry::list_sessions() const {
    std::shared_lock lock(m_sessions_mutex);
    std::vector<std::string> result;
    result.reserve(m_sessions.size());
    for (const auto & kv : m_sessions) {
        result.push_back(kv.first);
    }
    return result;
}

//
// Session-scoped execution
//

void tool_registry::ensure_executor_for_session(tool_session & session,
                                                 const std::string & executor_name) {
    if (session.executor_started.count(executor_name) && session.executor_started[executor_name]) {
        return;
    }

    // Needs m_executors_mutex to be held (shared) by the caller
    auto it = m_executors.find(executor_name);
    if (it == m_executors.end()) {
        return;
    }

    auto * se = dynamic_cast<tool_session_executor *>(it->second.get());
    if (se) {
        se->on_session_start(session.session_id, session.context);
    }

    session.executor_started[executor_name] = true;
}

tool_call_result tool_registry::execute_tool_in_session(const std::string & session_id,
                                                         const tool_call_request & request) {
    tool_call_result error;
    error.call_id = request.call_id;
    error.success = false;

    if (m_shutdown_requested.load()) {
        error.error_message = "Server is shutting down";
        error.error_code    = 503;
        return error;
    }

    // Resolve tool and executor
    std::string executor_name;
    tool_executor * exec_ptr = nullptr;
    {
        std::shared_lock tlock(m_tools_mutex);
        auto it = m_tools.find(request.tool_name);
        if (it == m_tools.end()) {
            error.error_message = "Tool not found: " + request.tool_name;
            error.error_code    = 404;
            return error;
        }
        exec_ptr      = it->second.executor;
        executor_name = it->second.executor_name;
    }

    if (!exec_ptr || !exec_ptr->is_healthy()) {
        error.error_message = "Executor unavailable for tool: " + request.tool_name;
        error.error_code    = exec_ptr ? 503 : 500;
        error.is_retryable  = (exec_ptr != nullptr);
        return error;
    }

    // Get / update session and ensure the executor is started for it
    {
        std::unique_lock slock(m_sessions_mutex);
        auto sit = m_sessions.find(session_id);
        if (sit == m_sessions.end()) {
            error.error_message = "Session not found: " + session_id;
            error.error_code    = 404;
            return error;
        }
        sit->second.touch();

        std::shared_lock elock(m_executors_mutex);
        ensure_executor_for_session(sit->second, executor_name);
    }

    // Execute (preferring session-aware path)
    auto * se = dynamic_cast<tool_session_executor *>(exec_ptr);

    auto t0 = std::chrono::steady_clock::now();
    tool_call_result result = se ? se->execute_in_session(session_id, request)
                                 : exec_ptr->execute(request);
    auto t1 = std::chrono::steady_clock::now();
    result.execution_time = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    return result;
}

//
// Cleanup
//

void tool_registry::cleanup_expired_sessions() {
    std::vector<std::string> expired;
    {
        std::shared_lock lock(m_sessions_mutex);
        for (const auto & kv : m_sessions) {
            if (kv.second.is_expired(m_session_ttl)) {
                expired.push_back(kv.first);
            }
        }
    }
    for (const auto & sid : expired) {
        LOG_INF("tool_registry: expiring session '%s'\n", sid.c_str());
        close_session(sid);
    }
}

//
// Health
//

tool_registry::health_status tool_registry::get_health() const {
    health_status status;

    {
        std::shared_lock elock(m_executors_mutex);
        bool all_healthy = !m_executors.empty();
        for (const auto & kv : m_executors) {
            bool h = kv.second->is_healthy();
            status.executor_health[kv.first] = h;
            if (!h) {
                all_healthy = false;
            }
        }
        status.healthy = all_healthy && m_initialized.load() && !m_shutdown_requested.load();
    }

    {
        std::shared_lock tlock(m_tools_mutex);
        status.registered_tools = m_tools.size();
    }

    {
        std::shared_lock slock(m_sessions_mutex);
        status.active_sessions = m_sessions.size();
    }

    return status;
}
