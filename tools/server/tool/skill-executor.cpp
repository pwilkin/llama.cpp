#include "skill-executor.h"

#include "log.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

skill_executor::skill_executor(config cfg) : m_config(std::move(cfg)) {}

std::string skill_executor::get_name() const {
    return m_name.empty() ? "skill:" + m_config.skill_path : m_name;
}

executor_capabilities skill_executor::get_capabilities() const {
    return executor_capabilities{}; // synchronous, no subprocess, no state
}

bool skill_executor::parse_skill_md() {
    fs::path skill_md_path = fs::path(m_config.skill_path) / "SKILL.md";
    std::ifstream f(skill_md_path);
    if (!f) {
        LOG_ERR("skill_executor: cannot open %s\n", skill_md_path.string().c_str());
        return false;
    }

    std::string line;

    // Expect opening ---
    if (!std::getline(f, line) || line != "---") {
        LOG_ERR("skill_executor: %s: missing frontmatter opening '---'\n", skill_md_path.string().c_str());
        return false;
    }

    // Parse frontmatter key: value pairs
    bool found_close = false;
    while (std::getline(f, line)) {
        if (line == "---") {
            found_close = true;
            break;
        }
        auto colon = line.find(':');
        if (colon == std::string::npos) {
            continue; // skip malformed or continuation lines
        }
        std::string key   = line.substr(0, colon);
        std::string value = line.substr(colon + 1);
        // Trim leading whitespace from value
        auto first_non_ws = value.find_first_not_of(" \t");
        value = (first_non_ws != std::string::npos) ? value.substr(first_non_ws) : std::string{};

        if (key == "name") {
            m_name = value;
        } else if (key == "description") {
            m_description = value;
        }
        // other frontmatter fields (license, compatibility, metadata, allowed-tools) are ignored
    }

    if (!found_close) {
        LOG_ERR("skill_executor: %s: missing frontmatter closing '---'\n", skill_md_path.string().c_str());
        return false;
    }
    if (m_name.empty()) {
        LOG_ERR("skill_executor: %s: missing required 'name' field in frontmatter\n", skill_md_path.string().c_str());
        return false;
    }

    // Read remaining content as the skill body (instructions)
    std::ostringstream oss;
    while (std::getline(f, line)) {
        oss << line << '\n';
    }
    m_body = oss.str();
    // Strip leading blank lines
    auto first_content = m_body.find_first_not_of("\n\r");
    if (first_content != std::string::npos) {
        m_body = m_body.substr(first_content);
    }

    return true;
}

bool skill_executor::initialize(const json & /*cfg*/) {
    if (!parse_skill_md()) {
        return false;
    }
    m_healthy = true;
    LOG_INF("skill_executor[%s]: loaded from %s\n", m_name.c_str(), m_config.skill_path.c_str());
    return true;
}

std::vector<tool_definition> skill_executor::list_provided_tools() const {
    if (!m_healthy) {
        return {};
    }

    // 1. Main skill tool — returns the body (agent instructions)
    tool_definition main_tool;
    main_tool.name        = m_name;
    main_tool.description = m_description.empty() ? "Skill: " + m_name : m_description;
    main_tool.schema      = json{
        {"type", "object"},
        {"properties", json::object()},
        {"additionalProperties", false}
    };

    // 2. Resources listing tool
    tool_definition resources_tool;
    resources_tool.name        = m_name + ":resources";
    resources_tool.description = "List all supplementary resource files available in the '" + m_name + "' skill "
                                 "(scripts, references, assets, etc.). Returns an array of relative paths. "
                                 "Use " + m_name + ":resource to retrieve a specific file.";
    resources_tool.schema      = json{
        {"type", "object"},
        {"properties", json::object()},
        {"additionalProperties", false}
    };

    // 3. Resource retrieval tool
    tool_definition resource_tool;
    resource_tool.name        = m_name + ":resource";
    resource_tool.description = "Read the contents of a resource file from the '" + m_name + "' skill. "
                                "Call " + m_name + ":resources first to discover available paths.";
    resource_tool.schema      = json{
        {"type", "object"},
        {"properties", json{
            {"resource_path", json{
                {"type", "string"},
                {"description", "Relative path to the resource from the skill root "
                                "(e.g. 'references/REFERENCE.md', 'scripts/extract.py')"}
            }}
        }},
        {"required", json::array({"resource_path"})},
        {"additionalProperties", false}
    };

    return {main_tool, resources_tool, resource_tool};
}

tool_call_result skill_executor::execute(const tool_call_request & request) {
    if (request.tool_name == m_name) {
        return handle_main(request);
    }
    if (request.tool_name == m_name + ":resources") {
        return handle_resources(request);
    }
    if (request.tool_name == m_name + ":resource") {
        return handle_resource(request);
    }

    tool_call_result r;
    r.call_id       = request.call_id;
    r.success       = false;
    r.error_message = "Unknown tool: " + request.tool_name;
    r.error_code    = 404;
    return r;
}

// Returns the skill body (instructions).
tool_call_result skill_executor::handle_main(const tool_call_request & request) const {
    tool_call_result r;
    r.call_id = request.call_id;
    r.success = true;
    r.output  = m_body;
    return r;
}

std::vector<std::string> skill_executor::list_resource_paths() const {
    std::vector<std::string> paths;
    fs::path root(m_config.skill_path);
    try {
        for (const auto & entry : fs::recursive_directory_iterator(root)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string rel = fs::relative(entry.path(), root).string();
            if (rel == "SKILL.md") {
                continue; // the skill definition itself is not a resource
            }
            paths.push_back(std::move(rel));
        }
    } catch (const std::exception & e) {
        LOG_WRN("skill_executor[%s]: list_resource_paths error: %s\n", m_name.c_str(), e.what());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// Returns a JSON array of relative resource paths.
tool_call_result skill_executor::handle_resources(const tool_call_request & request) const {
    tool_call_result r;
    r.call_id = request.call_id;
    auto paths = list_resource_paths();
    json arr = json::array();
    for (const auto & p : paths) {
        arr.push_back(p);
    }
    r.output  = std::move(arr);
    r.success = true;
    return r;
}

// Reads a specific resource file, with path-traversal protection.
tool_call_result skill_executor::handle_resource(const tool_call_request & request) const {
    tool_call_result r;
    r.call_id = request.call_id;

    std::string resource_path = request.arguments.value("resource_path", "");
    if (resource_path.empty()) {
        r.success       = false;
        r.error_message = "Missing required argument 'resource_path'";
        r.error_code    = 400;
        return r;
    }

    // Security: prevent path traversal outside the skill root
    fs::path root(m_config.skill_path);
    fs::path root_canonical;
    fs::path target;
    try {
        root_canonical = fs::weakly_canonical(root);
        target         = fs::weakly_canonical(root / resource_path);
    } catch (const std::exception & e) {
        r.success       = false;
        r.error_message = std::string("Invalid path: ") + e.what();
        r.error_code    = 400;
        return r;
    }

    // Check that every component of the relative path stays within root
    fs::path rel = fs::relative(target, root_canonical);
    for (const auto & part : rel) {
        if (part.string() == "..") {
            r.success       = false;
            r.error_message = "Path traversal not allowed";
            r.error_code    = 400;
            return r;
        }
    }

    if (!fs::exists(target) || !fs::is_regular_file(target)) {
        r.success       = false;
        r.error_message = "Resource not found: " + resource_path;
        r.error_code    = 404;
        return r;
    }

    std::ifstream f(target);
    if (!f) {
        r.success       = false;
        r.error_message = "Cannot read resource: " + resource_path;
        r.error_code    = 500;
        return r;
    }

    std::ostringstream oss;
    oss << f.rdbuf();
    r.output  = oss.str();
    r.success = true;
    return r;
}
