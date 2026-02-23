#pragma once

#include "tool-executor.h"

#include <string>

/**
 * Extended executor interface for executors that maintain per-session state
 * (e.g., MCP servers, plugin processes with conversation context).
 *
 * Lifecycle per session:
 *   on_session_start()  → session created
 *   execute()           → one or more calls
 *   on_session_end()    → session closed or expired
 */
class tool_session_executor : public tool_executor {
public:
    ~tool_session_executor() override = default;

    /**
     * Called when a new session is created.
     * @param session_id  unique session identifier
     * @param context     arbitrary JSON context provided by the client
     * @return true on success
     */
    virtual bool on_session_start(const std::string & session_id, const json & context) = 0;

    /**
     * Called when a session ends (explicit close or expiry).
     * @param session_id  identifier of the ending session
     */
    virtual void on_session_end(const std::string & session_id) = 0;

    /**
     * Execute a tool call scoped to a session.
     *
     * Implementations may use session_id to retrieve per-session state.
     * Defaults to the non-session execute() if not overridden.
     */
    virtual tool_call_result execute_in_session(const std::string & session_id,
                                                const tool_call_request & request) {
        (void)session_id;
        return execute(request);
    }
};
