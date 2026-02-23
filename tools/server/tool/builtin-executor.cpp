#include "builtin-executor.h"

#include "log.h"

#include <chrono>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>

//
// Simple arithmetic expression parser
//

struct expr_parser {
    const std::string & s;
    size_t pos;

    expr_parser(const std::string & str) : s(str), pos(0) {}

    void skip_ws() {
        while (pos < s.size() && isspace((unsigned char)s[pos])) {
            pos++;
        }
    }

    double parse_number() {
        skip_ws();
        double val = 0;
        bool has_digits = false;
        while (pos < s.size() && isdigit((unsigned char)s[pos])) {
            val = val * 10 + (s[pos] - '0');
            pos++;
            has_digits = true;
        }
        if (!has_digits) {
            throw std::runtime_error("expected number at position " + std::to_string(pos));
        }
        if (pos < s.size() && s[pos] == '.') {
            pos++;
            double frac = 0.1;
            while (pos < s.size() && isdigit((unsigned char)s[pos])) {
                val += (s[pos] - '0') * frac;
                frac *= 0.1;
                pos++;
            }
        }
        return val;
    }

    double parse_factor() {
        skip_ws();
        if (pos < s.size() && s[pos] == '(') {
            pos++;
            double val = parse_expr();
            skip_ws();
            if (pos >= s.size() || s[pos] != ')') {
                throw std::runtime_error("expected ')' at position " + std::to_string(pos));
            }
            pos++;
            return val;
        }
        if (pos < s.size() && s[pos] == '-') {
            pos++;
            return -parse_factor();
        }
        if (pos < s.size() && s[pos] == '+') {
            pos++;
            return parse_factor();
        }
        return parse_number();
    }

    double parse_term() {
        double val = parse_factor();
        skip_ws();
        while (pos < s.size() && (s[pos] == '*' || s[pos] == '/')) {
            char op = s[pos++];
            double rhs = parse_factor();
            if (op == '*') {
                val *= rhs;
            } else {
                if (rhs == 0.0) {
                    throw std::runtime_error("division by zero");
                }
                val /= rhs;
            }
            skip_ws();
        }
        return val;
    }

    double parse_expr() {
        double val = parse_term();
        skip_ws();
        while (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) {
            char op = s[pos++];
            double rhs = parse_term();
            if (op == '+') {
                val += rhs;
            } else {
                val -= rhs;
            }
            skip_ws();
        }
        return val;
    }

    double evaluate() {
        double val = parse_expr();
        skip_ws();
        if (pos != s.size()) {
            throw std::runtime_error("unexpected character '" + std::string(1, s[pos])
                + "' at position " + std::to_string(pos));
        }
        return val;
    }
};

//
// Built-in tool implementations
//

tool_call_result builtin_tools::datetime_tool(const tool_call_request & req) {
    tool_call_result result;
    result.call_id = req.call_id;

    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &time_t_now);
#else
    gmtime_r(&time_t_now, &tm_utc);
#endif

    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

    result.success = true;
    result.output  = json{{"timestamp", std::string(buf)}, {"timezone", "UTC"}};
    return result;
}

tool_call_result builtin_tools::calculator(const tool_call_request & req) {
    tool_call_result result;
    result.call_id = req.call_id;

    if (!req.arguments.contains("expression") || !req.arguments["expression"].is_string()) {
        result.error_message = "Missing or invalid 'expression' field (must be a string)";
        result.error_code    = 400;
        return result;
    }

    std::string expr = req.arguments["expression"].get<std::string>();

    try {
        expr_parser parser(expr);
        double val = parser.evaluate();
        result.success = true;
        result.output  = json{{"result", val}, {"expression", expr}};
    } catch (const std::exception & e) {
        result.error_message = std::string("Expression error: ") + e.what();
        result.error_code    = 400;
    }

    return result;
}

//
// builtin_executor
//

builtin_executor::builtin_executor() {
    // datetime
    tool_definition dt;
    dt.name        = "datetime";
    dt.description = "Returns the current UTC date and time as an ISO 8601 string.";
    dt.schema      = json{{"type", "object"}, {"properties", json::object()}, {"additionalProperties", false}};
    register_tool(dt, builtin_tools::datetime_tool);

    // calculator
    tool_definition calc;
    calc.name        = "calculator";
    calc.description = "Evaluates a basic arithmetic expression (+, -, *, /, parentheses).";
    calc.schema      = json{
        {"type", "object"},
        {"properties", json{
            {"expression", json{{"type", "string"}, {"description", "Arithmetic expression to evaluate, e.g. '2 * (3 + 4)'"}}}
        }},
        {"required", json::array({"expression"})},
        {"additionalProperties", false}
    };
    register_tool(calc, builtin_tools::calculator);
}

void builtin_executor::register_tool(const tool_definition & def, builtin_tool_func func) {
    m_definitions[def.name] = def;
    m_functions[def.name]   = std::move(func);
}

executor_capabilities builtin_executor::get_capabilities() const {
    return executor_capabilities{false, false, false, false};
}

std::vector<tool_definition> builtin_executor::list_provided_tools() const {
    std::vector<tool_definition> result;
    result.reserve(m_definitions.size());
    for (const auto & kv : m_definitions) {
        result.push_back(kv.second);
    }
    return result;
}

bool builtin_executor::initialize(const json & /*config*/) {
    m_initialized = true;
    return true;
}

void builtin_executor::shutdown() {
    m_initialized = false;
}

tool_call_result builtin_executor::execute(const tool_call_request & request) {
    tool_call_result error;
    error.call_id = request.call_id;
    error.success = false;

    auto it = m_functions.find(request.tool_name);
    if (it == m_functions.end()) {
        error.error_message = "Unknown built-in tool: " + request.tool_name;
        error.error_code    = 404;
        return error;
    }

    try {
        return it->second(request);
    } catch (const std::exception & e) {
        error.error_message = std::string("Unexpected error: ") + e.what();
        error.error_code    = 500;
        return error;
    }
}
