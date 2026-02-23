#pragma once

#include "tool-executor.h"

#include <functional>
#include <map>

/**
 * Built-in tool function signature.
 */
using builtin_tool_func = std::function<tool_call_result(const tool_call_request &)>;

/**
 * Built-in executor for native C++ tools.
 *
 * Runs in-process, synchronously, with no persistent state between calls.
 * Ideal for simple utilities (datetime, calculator, etc.).
 */
class builtin_executor : public tool_executor {
public:
    builtin_executor();
    ~builtin_executor() override = default;

    // Register a tool with a handler function
    void register_tool(const tool_definition & def, builtin_tool_func func);

    // tool_executor interface
    std::string get_name() const override { return "builtin"; }
    executor_capabilities get_capabilities() const override;
    std::vector<tool_definition> list_provided_tools() const override;
    bool initialize(const json & config) override;
    void shutdown() override;
    bool is_healthy() const override { return m_initialized; }
    tool_call_result execute(const tool_call_request & request) override;

private:
    std::map<std::string, tool_definition> m_definitions;
    std::map<std::string, builtin_tool_func> m_functions;
    bool m_initialized = false;
};

/**
 * Built-in tool implementations.
 */
namespace builtin_tools {
    tool_call_result datetime_tool(const tool_call_request & req);
    tool_call_result calculator(const tool_call_request & req);
}
