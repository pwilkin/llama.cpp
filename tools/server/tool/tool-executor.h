#pragma once

#include "tool.h"

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

/**
 * Execution mode for a tool.
 */
enum class tool_execution_mode {
    SYNC,       // Synchronous execution (built-in functions)
    ASYNC,      // Asynchronous with callback
    STREAMING   // Streaming results (for long-running tools)
};

/**
 * Executor capability flags.
 */
struct executor_capabilities {
    bool supports_state        = false; // Maintains state between calls
    bool supports_streaming    = false; // Can stream partial results
    bool supports_cancellation = false; // Can cancel in-progress calls
    bool requires_subprocess   = false; // Needs isolated process
};

/**
 * Callback for streaming results.
 */
using tool_stream_callback = std::function<void(const json & partial_result, bool done)>;

/**
 * Abstract base class for all tool executors.
 */
class tool_executor {
public:
    virtual ~tool_executor() = default;

    // Core properties
    virtual std::string get_name() const = 0;
    virtual executor_capabilities get_capabilities() const = 0;

    // Return all tools this executor provides (called by registry after initialize)
    virtual std::vector<tool_definition> list_provided_tools() const { return {}; }

    // Lifecycle
    virtual bool initialize(const json & config) = 0;
    virtual void shutdown() = 0;
    virtual bool is_healthy() const = 0;

    // Synchronous execution
    virtual tool_call_result execute(const tool_call_request & request) = 0;

    // Async execution — default launches execute() in a thread
    virtual std::future<tool_call_result> execute_async(const tool_call_request & request);

    // Streaming execution — default calls execute() and fires callback once with done=true
    virtual bool execute_streaming(const tool_call_request & request, const tool_stream_callback & callback);

    // Cancellation — default: not supported
    virtual bool cancel(const std::string & call_id);
};

using tool_executor_ptr = std::unique_ptr<tool_executor>;
