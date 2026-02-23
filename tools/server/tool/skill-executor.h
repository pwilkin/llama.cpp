#pragma once

#include "tool-executor.h"

#include <string>
#include <vector>

/**
 * Executor for a single Agent Skill (https://agentskills.io/specification).
 *
 * Parses a skill directory containing a SKILL.md file with YAML frontmatter
 * and registers three tools:
 *
 *   <name>           – returns the skill body (instructions for the agent)
 *   <name>:resources – lists all supplementary files in the skill directory
 *   <name>:resource  – reads a specific file by relative path
 *
 * This follows the Agent Skills progressive-disclosure model: the agent first
 * activates the skill to receive its instructions, then loads resources on demand.
 */
class skill_executor : public tool_executor {
public:
    struct config {
        std::string skill_path; // path to the skill root directory (contains SKILL.md)
    };

    explicit skill_executor(config cfg);

    std::string              get_name()             const override;
    executor_capabilities    get_capabilities()     const override;
    std::vector<tool_definition> list_provided_tools() const override;

    bool initialize(const json & cfg) override;
    void shutdown()                   override {}
    bool is_healthy()                 const override { return m_healthy; }

    tool_call_result execute(const tool_call_request & request) override;

private:
    config      m_config;
    std::string m_name;
    std::string m_description;
    std::string m_body;       // SKILL.md content after frontmatter
    bool        m_healthy = false;

    bool parse_skill_md();
    std::vector<std::string> list_resource_paths() const;

    tool_call_result handle_main      (const tool_call_request & request) const;
    tool_call_result handle_resources (const tool_call_request & request) const;
    tool_call_result handle_resource  (const tool_call_request & request) const;
};
