#include "plugin-executor.h"

#include "log.h"

#include <chrono>
#include <stdexcept>

plugin_executor::plugin_executor(config cfg) : m_config(std::move(cfg)) {}

plugin_executor::~plugin_executor() {
    shutdown();
}

executor_capabilities plugin_executor::get_capabilities() const {
    return executor_capabilities{false, false, false, true /*requires_subprocess*/};
}

bool plugin_executor::initialize(const json & /*cfg*/) {
    std::vector<std::string> argv;
    argv.push_back(m_config.command);
    for (const auto & a : m_config.args) {
        argv.push_back(a);
    }

    bool started = m_proc.spawn(argv, m_config.env,
        [this](const std::string & line) { on_line(line); });

    if (!started) {
        LOG_ERR("plugin_executor[%s]: failed to spawn process\n", m_config.name.c_str());
        return false;
    }

    // Discover tools via the initialize RPC
    try {
        json resp = send_rpc("initialize", json::object(), m_config.init_timeout_ms);

        if (resp.contains("result") && resp["result"].contains("tools")) {
            for (const auto & t : resp["result"]["tools"]) {
                tool_definition def;
                def.name        = t.value("name", "");
                def.description = t.value("description", "");
                if (t.contains("schema")) {
                    def.schema = t["schema"];
                }
                if (!def.name.empty()) {
                    m_tools.push_back(std::move(def));
                }
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("plugin_executor[%s]: initialize RPC failed: %s\n",
                m_config.name.c_str(), e.what());
        m_proc.stop();
        return false;
    }

    m_healthy.store(true);
    LOG_INF("plugin_executor[%s]: ready with %zu tools\n",
            m_config.name.c_str(), m_tools.size());
    return true;
}

void plugin_executor::shutdown() {
    m_healthy.store(false);

    // Stop the subprocess — this joins the reader thread, so no more on_line() calls
    m_proc.stop();

    // Fail any pending requests
    std::lock_guard lock(m_rpc_mutex);
    for (auto & kv : m_pending) {
        try {
            kv.second.set_exception(std::make_exception_ptr(
                std::runtime_error("plugin_executor shut down")));
        } catch (...) {}
    }
    m_pending.clear();
}

tool_call_result plugin_executor::execute(const tool_call_request & request) {
    tool_call_result result;
    result.call_id = request.call_id;
    result.success = false;

    if (!m_healthy.load()) {
        result.error_message = "Plugin executor is not healthy";
        result.error_code    = 503;
        result.is_retryable  = true;
        return result;
    }

    json params = {{"name", request.tool_name}, {"arguments", request.arguments}};

    try {
        json resp = send_rpc("call_tool", params, m_config.call_timeout_ms);

        if (resp.contains("error")) {
            result.error_message = resp["error"].value("message", "unknown plugin error");
            result.error_code    = resp["error"].value("code", 500);
        } else if (resp.contains("result")) {
            const auto & r  = resp["result"];
            result.success   = r.value("success", true);
            result.output    = r.contains("output") ? r["output"] : r;
            if (!result.success && r.contains("error")) {
                result.error_message = r["error"];
            }
        }
    } catch (const std::exception & e) {
        result.error_message = std::string("RPC error: ") + e.what();
        result.error_code    = 500;
    }

    return result;
}

bool plugin_executor::on_session_start(const std::string & /*session_id*/, const json & /*context*/) {
    return true; // single shared process — no per-session setup
}

void plugin_executor::on_session_end(const std::string & /*session_id*/) {
    // nothing
}

json plugin_executor::send_rpc(const std::string & method, const json & params, int timeout_ms) {
    int id = m_next_id.fetch_add(1, std::memory_order_relaxed);

    std::future<json> fut;
    {
        std::lock_guard lock(m_rpc_mutex);
        auto & prom = m_pending[id]; // default-construct in-place
        fut = prom.get_future();
    }

    json req = {{"method", method}, {"params", params}, {"id", id}};
    if (!m_proc.write_stdin(req.dump())) {
        std::lock_guard lock(m_rpc_mutex);
        m_pending.erase(id);
        throw std::runtime_error("Failed to write to plugin stdin");
    }

    if (fut.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::timeout) {
        std::lock_guard lock(m_rpc_mutex);
        m_pending.erase(id);
        throw std::runtime_error("Plugin RPC timed out (method=" + method + ")");
    }

    return fut.get();
}

void plugin_executor::on_line(const std::string & line) {
    if (line.empty()) {
        return;
    }
    try {
        json msg = json::parse(line);
        if (msg.contains("id") && msg["id"].is_number_integer()) {
            int id = msg["id"].get<int>();
            std::lock_guard lock(m_rpc_mutex);
            auto it = m_pending.find(id);
            if (it != m_pending.end()) {
                it->second.set_value(std::move(msg));
                m_pending.erase(it);
            }
        }
    } catch (const std::exception & e) {
        LOG_WRN("plugin_executor[%s]: failed to parse output: %s\n",
                m_config.name.c_str(), e.what());
    }
}
