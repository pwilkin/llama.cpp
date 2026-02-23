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
 * Executor for MCP (Model Context Protocol) servers.
 *
 * Implements the MCP stdio transport (JSON-RPC 2.0 over stdin/stdout).
 *
 * Handshake:
 *   → {"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05",
 *       "capabilities":{},"clientInfo":{"name":"llama-server","version":"1.0"}},"id":1}
 *   ← {"jsonrpc":"2.0","result":{"protocolVersion":...,"capabilities":...},"id":1}
 *   → {"jsonrpc":"2.0","method":"notifications/initialized","params":{}}   (no id)
 *   → {"jsonrpc":"2.0","method":"tools/list","params":{},"id":2}
 *   ← {"jsonrpc":"2.0","result":{"tools":[...]},"id":2}
 *
 * Tool call:
 *   → {"jsonrpc":"2.0","method":"tools/call","params":{"name":"...","arguments":{...}},"id":N}
 *   ← {"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"..."}],"isError":false},"id":N}
 */
class mcp_executor : public tool_session_executor {
public:
    struct config {
        std::string              name = "mcp";
        std::string              command;
        std::vector<std::string> args;
        std::vector<std::string> env;
        int                      init_timeout_ms = 10000;
        int                      call_timeout_ms = 30000;
    };

    explicit mcp_executor(config cfg);
    ~mcp_executor() override;

    // tool_executor interface
    std::string get_name() const override { return m_config.name; }
    executor_capabilities get_capabilities() const override;
    std::vector<tool_definition> list_provided_tools() const override { return m_tools; }
    bool initialize(const json & cfg) override;
    void shutdown() override;
    bool is_healthy() const override { return m_healthy.load(); }
    tool_call_result execute(const tool_call_request & request) override;

    // tool_session_executor interface
    bool on_session_start(const std::string & session_id, const json & context) override;
    void on_session_end(const std::string & session_id) override;

private:
    json send_request(const std::string & method, const json & params, int timeout_ms);
    void send_notification(const std::string & method, const json & params);
    void on_line(const std::string & line);

    // Convert MCP content array to a single json output value
    static json mcp_content_to_output(const json & content);

    config                              m_config;
    subprocess_handle                   m_proc;

    std::mutex                          m_rpc_mutex;
    std::map<int, std::promise<json>>   m_pending;
    std::atomic<int>                    m_next_id{1};

    std::vector<tool_definition>        m_tools;
    std::atomic<bool>                   m_healthy{false};
};
