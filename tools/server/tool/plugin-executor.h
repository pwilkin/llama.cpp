#pragma once

#include "tool-session-executor.h"
#include "subprocess-manager.h"

#include <atomic>
#include <future>
#include <map>
#include <mutex>
#include <string>
#include <vector>

/**
 * Executor for external plugin tools.
 *
 * Communicates via a simple line-delimited JSON protocol over stdin/stdout:
 *
 *   → {"method": "initialize", "params": {}, "id": 1}
 *   ← {"result": {"tools": [{"name":..., "description":..., "schema":...}]}, "id": 1}
 *
 *   → {"method": "call_tool", "params": {"name": "...", "arguments": {...}}, "id": 2}
 *   ← {"result": {"output": ..., "success": true}, "id": 2}
 *   ← {"error": {"message": "...", "code": 400}, "id": 2}
 */
class plugin_executor : public tool_session_executor {
public:
    struct config {
        std::string              name = "plugin";
        std::string              command;
        std::vector<std::string> args;
        std::vector<std::string> env;  // "KEY=VALUE" entries; empty = inherit
        int                      init_timeout_ms = 5000;
        int                      call_timeout_ms = 30000;
    };

    explicit plugin_executor(config cfg);
    ~plugin_executor() override;

    // tool_executor interface
    std::string get_name() const override { return m_config.name; }
    executor_capabilities get_capabilities() const override;
    std::vector<tool_definition> list_provided_tools() const override { return m_tools; }
    bool initialize(const json & cfg) override;
    void shutdown() override;
    bool is_healthy() const override { return m_healthy.load(); }
    tool_call_result execute(const tool_call_request & request) override;

    // tool_session_executor interface (no-op: single shared process per executor)
    bool on_session_start(const std::string & session_id, const json & context) override;
    void on_session_end(const std::string & session_id) override;

private:
    json send_rpc(const std::string & method, const json & params, int timeout_ms);
    void on_line(const std::string & line);

    config                              m_config;
    subprocess_handle                   m_proc;

    std::mutex                          m_rpc_mutex;
    std::map<int, std::promise<json>>   m_pending;
    std::atomic<int>                    m_next_id{1};

    std::vector<tool_definition>        m_tools;
    std::atomic<bool>                   m_healthy{false};
};
