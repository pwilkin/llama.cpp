#include "tool-executor.h"

#include <future>

std::future<tool_call_result> tool_executor::execute_async(const tool_call_request & request) {
    return std::async(std::launch::async, [this, request]() {
        return this->execute(request);
    });
}

bool tool_executor::execute_streaming(const tool_call_request & request, const tool_stream_callback & callback) {
    if (!get_capabilities().supports_streaming) {
        return false;
    }
    auto result = execute(request);
    callback(result.output, true);
    return result.success;
}

bool tool_executor::cancel(const std::string & /*call_id*/) {
    return false;
}
