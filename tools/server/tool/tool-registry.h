#pragma once

#include "tool.h"
#include "tool-executor.h"

#include <atomic>
#include <chrono>
#include <future>
#include <map>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * Links a tool definition to the executor that handles it.
 */
struct tool_registration {
    tool_definition   definition;
    tool_executor   * executor;      // non-owning; lifetime managed by registry
    std::string       executor_name;
};

/**
 * Per-session state.  Tracks which executors are active for the session
 * and arbitrary client-provided context.
 */
struct tool_session {
    std::string                       session_id;
    json                              context;
    std::map<std::string, bool>       executor_started; // executor_name → started?
    std::chrono::steady_clock::time_point last_used;

    tool_session() : last_used(std::chrono::steady_clock::now()) {}

    void touch() { last_used = std::chrono::steady_clock::now(); }

    bool is_expired(std::chrono::seconds ttl) const {
        return std::chrono::steady_clock::now() - last_used > ttl;
    }
};

/**
 * Central registry that manages tool executors, tools, and sessions.
 *
 * Thread-safe via separate shared_mutex for tools, executors, and sessions.
 */
class tool_registry {
public:
    tool_registry();
    ~tool_registry();

    tool_registry(const tool_registry &)            = delete;
    tool_registry & operator=(const tool_registry &) = delete;

    // Lifecycle
    bool initialize(const json & config);
    void shutdown();
    bool is_initialized() const { return m_initialized.load(); }

    // Executor management
    // Takes ownership; calls initialize() then auto-registers list_provided_tools()
    void register_executor(tool_executor_ptr executor);
    void unregister_executor(const std::string & name);
    tool_executor * get_executor(const std::string & name) const;

    // Tool management
    void register_tool(const tool_definition & def, const std::string & executor_name);
    void unregister_tool(const std::string & name);

    std::vector<tool_definition> list_tools() const;
    std::optional<tool_definition> get_tool(const std::string & name) const;
    bool has_tool(const std::string & name) const;

    // Stateless execution
    tool_call_result execute_tool(const tool_call_request & request);
    std::future<tool_call_result> execute_tool_async(const tool_call_request & request);

    // Session management
    std::string create_session(const json & context = json::object());
    bool has_session(const std::string & session_id) const;
    bool close_session(const std::string & session_id);
    std::vector<std::string> list_sessions() const;

    // Session-scoped execution
    tool_call_result execute_tool_in_session(const std::string & session_id,
                                             const tool_call_request & request);

    // Health
    struct health_status {
        bool healthy = false;
        std::map<std::string, bool> executor_health;
        size_t registered_tools    = 0;
        size_t active_sessions     = 0;
    };
    health_status get_health() const;

    void request_shutdown() { m_shutdown_requested.store(true); }
    bool is_shutting_down() const { return m_shutdown_requested.load(); }

private:
    void cleanup_expired_sessions();
    void ensure_executor_for_session(tool_session & session, const std::string & executor_name);

    mutable std::shared_mutex m_tools_mutex;
    std::map<std::string, tool_registration> m_tools;

    mutable std::shared_mutex m_executors_mutex;
    std::map<std::string, tool_executor_ptr> m_executors;

    mutable std::shared_mutex m_sessions_mutex;
    std::map<std::string, tool_session> m_sessions;
    std::chrono::seconds m_session_ttl{300};

    // Background cleanup thread
    std::thread m_cleanup_thread;

    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_shutdown_requested{false};
};
