#include "mcp-executor.h"

#include "log.h"

#include <chrono>
#include <stdexcept>

mcp_executor::mcp_executor(config cfg) : m_config(std::move(cfg)) {}

mcp_executor::~mcp_executor() {
    shutdown();
}

executor_capabilities mcp_executor::get_capabilities() const {
    return executor_capabilities{false, false, false, true /*requires_subprocess*/};
}

bool mcp_executor::initialize(const json & /*cfg*/) {
    std::vector<std::string> argv;
    argv.push_back(m_config.command);
    for (const auto & a : m_config.args) {
        argv.push_back(a);
    }

    bool started = m_proc.spawn(argv, m_config.env,
        [this](const std::string & line) { on_line(line); });

    if (!started) {
        LOG_ERR("mcp_executor[%s]: failed to spawn process\n", m_config.name.c_str());
        return false;
    }

    // MCP handshake: initialize
    try {
        json init_params = {
            {"protocolVersion", "2024-11-05"},
            {"capabilities",    json::object()},
            {"clientInfo",      json{{"name", "llama-server"}, {"version", "1.0"}}}
        };
        json resp = send_request("initialize", init_params, m_config.init_timeout_ms);

        if (resp.contains("error")) {
            LOG_ERR("mcp_executor[%s]: initialize failed: %s\n",
                    m_config.name.c_str(), resp["error"].dump().c_str());
            m_proc.stop();
            return false;
        }
    } catch (const std::exception & e) {
        LOG_ERR("mcp_executor[%s]: initialize RPC error: %s\n",
                m_config.name.c_str(), e.what());
        m_proc.stop();
        return false;
    }

    // Notify server that client is initialized (fire-and-forget)
    send_notification("notifications/initialized", json::object());

    // Discover tools
    try {
        json resp = send_request("tools/list", json::object(), m_config.init_timeout_ms);

        if (resp.contains("result") && resp["result"].contains("tools")) {
            for (const auto & t : resp["result"]["tools"]) {
                tool_definition def;
                def.name        = t.value("name", "");
                def.description = t.value("description", "");
                // MCP schema is in inputSchema
                if (t.contains("inputSchema")) {
                    def.schema = t["inputSchema"];
                }
                if (!def.name.empty()) {
                    m_tools.push_back(std::move(def));
                }
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("mcp_executor[%s]: tools/list RPC error: %s\n",
                m_config.name.c_str(), e.what());
        m_proc.stop();
        return false;
    }

    m_healthy.store(true);
    LOG_INF("mcp_executor[%s]: ready with %zu tools\n",
            m_config.name.c_str(), m_tools.size());
    return true;
}

void mcp_executor::shutdown() {
    m_healthy.store(false);

    // Stop the subprocess — joins reader thread, so no more on_line() calls
    m_proc.stop();

    // Fail any pending requests
    std::lock_guard lock(m_rpc_mutex);
    for (auto & kv : m_pending) {
        try {
            kv.second.set_exception(std::make_exception_ptr(
                std::runtime_error("mcp_executor shut down")));
        } catch (const std::future_error &) {
            // Promise was already satisfied (response arrived during shutdown)
        }
    }
    m_pending.clear();
}

tool_call_result mcp_executor::execute(const tool_call_request & request) {
    tool_call_result result;
    result.call_id = request.call_id;
    result.success = false;

    if (!m_healthy.load()) {
        result.error_message = "MCP executor is not healthy";
        result.error_code    = 503;
        result.is_retryable  = true;
        return result;
    }

    json params = {{"name", request.tool_name}, {"arguments", request.arguments}};

    try {
        json resp = send_request("tools/call", params, m_config.call_timeout_ms);

        if (resp.contains("error")) {
            const auto & err = resp["error"];
            result.error_message = err.value("message", "unknown MCP error");
            result.error_code    = err.value("code", 500);
        } else if (resp.contains("result")) {
            const auto & r = resp["result"];
            bool is_error  = r.value("isError", false);
            result.success = !is_error;
            if (r.contains("content")) {
                result.output = mcp_content_to_output(r["content"]);
            }
            if (is_error) {
                result.error_message = result.output.is_string()
                                           ? result.output.get<std::string>()
                                           : result.output.dump();
                result.error_code = 500;
            }
        }
    } catch (const std::exception & e) {
        result.error_message = std::string("MCP RPC error: ") + e.what();
        result.error_code    = 500;
    }

    return result;
}

bool mcp_executor::on_session_start(const std::string & /*session_id*/, const json & /*context*/) {
    return true; // single shared process — no per-session MCP sessions in this implementation
}

void mcp_executor::on_session_end(const std::string & /*session_id*/) {
    // nothing
}

json mcp_executor::send_request(const std::string & method, const json & params, int timeout_ms) {
    int id = m_next_id.fetch_add(1, std::memory_order_relaxed);

    std::future<json> fut;
    {
        std::lock_guard lock(m_rpc_mutex);
        auto & prom = m_pending[id];
        fut = prom.get_future();
    }

    json req = {
        {"jsonrpc", "2.0"},
        {"method",  method},
        {"params",  params},
        {"id",      id}
    };

    if (!m_proc.write_stdin(req.dump())) {
        std::lock_guard lock(m_rpc_mutex);
        m_pending.erase(id);
        throw std::runtime_error("Failed to write to MCP server stdin");
    }

    if (fut.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::timeout) {
        std::lock_guard lock(m_rpc_mutex);
        m_pending.erase(id);
        throw std::runtime_error("MCP request timed out (method=" + method + ")");
    }

    return fut.get();
}

void mcp_executor::send_notification(const std::string & method, const json & params) {
    json notif = {{"jsonrpc", "2.0"}, {"method", method}, {"params", params}};
    m_proc.write_stdin(notif.dump()); // fire and forget
}

void mcp_executor::on_line(const std::string & line) {
    if (line.empty()) {
        return;
    }
    try {
        json msg = json::parse(line);
        // Only dispatch responses (have an integer id); ignore notifications
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
        LOG_WRN("mcp_executor[%s]: failed to parse output: %s\n",
                m_config.name.c_str(), e.what());
    }
}

json mcp_executor::mcp_content_to_output(const json & content) {
    if (!content.is_array() || content.empty()) {
        return content;
    }
    // If there's a single text item, return just the text string
    if (content.size() == 1 && content[0].value("type", "") == "text") {
        return content[0].value("text", "");
    }
    // Otherwise collect all text items into an array
    json parts = json::array();
    for (const auto & item : content) {
        std::string type = item.value("type", "");
        if (type == "text") {
            parts.push_back(item.value("text", ""));
        } else {
            parts.push_back(item); // pass through non-text content as-is
        }
    }
    return parts;
}
