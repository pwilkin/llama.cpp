#pragma once

#include <nlohmann/json.hpp>
#include <chrono>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

/**
 * Tool input schema flexibility levels.
 */
enum class tool_input_schema_type {
    SCHEMA_STRICT,  // Standard JSON Schema (OpenAI style)
    SCHEMA_FREE,    // Single blob input (Responses API freeform)
    SCHEMA_CUSTOM   // Executor-specific format
};

/**
 * Tool definition — metadata about an available tool.
 */
struct tool_definition {
    std::string name;
    std::string description;
    tool_input_schema_type schema_type = tool_input_schema_type::SCHEMA_STRICT;
    json schema;                         // JSON Schema for SCHEMA_STRICT, optional hints otherwise
    std::vector<std::string> tags;
    std::string version;
    json extras;
};

/**
 * Tool call request — represents an invocation.
 */
struct tool_call_request {
    std::string call_id;
    std::string tool_name;
    json arguments;
    json context;
    std::chrono::steady_clock::time_point created_at;
};

/**
 * Tool call result — represents an outcome.
 */
struct tool_call_result {
    std::string call_id;
    bool success = false;
    json output;
    std::string error_message;
    int error_code = 0;
    std::chrono::milliseconds execution_time{0};
    bool is_retryable = false;
};
