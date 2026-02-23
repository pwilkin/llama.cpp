#include "subprocess-manager.h"

#include "log.h"

#include <chrono>
#include <cstdio>
#include <cstring>

#ifndef _WIN32
#  include <signal.h>
#endif

subprocess_handle::subprocess_handle() = default;

subprocess_handle::~subprocess_handle() {
    if (m_spawned.load()) {
        stop(2000);
    }
}

bool subprocess_handle::spawn(const std::vector<std::string> & argv,
                               const std::vector<std::string> & env,
                               output_cb on_output) {
    if (m_spawned.load()) {
        LOG_ERR("subprocess_handle: already spawned\n");
        return false;
    }
    if (argv.empty()) {
        LOG_ERR("subprocess_handle: empty argv\n");
        return false;
    }

    // Build null-terminated C arrays (matches server-models.cpp pattern)
    std::vector<const char *> c_argv;
    c_argv.reserve(argv.size() + 1);
    for (const auto & a : argv) {
        c_argv.push_back(a.c_str());
    }
    c_argv.push_back(nullptr);

    std::vector<const char *> c_env;
    const char * const * env_ptr = nullptr;
    if (!env.empty()) {
        c_env.reserve(env.size() + 1);
        for (const auto & e : env) {
            c_env.push_back(e.c_str());
        }
        c_env.push_back(nullptr);
        env_ptr = c_env.data();
    }

    int options = subprocess_option_no_window
                | subprocess_option_combined_stdout_stderr;

    int ret = subprocess_create_ex(c_argv.data(), options, env_ptr, &m_proc);
    if (ret != 0) {
        LOG_ERR("subprocess_handle: subprocess_create_ex failed (ret=%d)\n", ret);
        return false;
    }

    m_spawned.store(true);

    // Start the reader thread
    m_reader = std::thread([this, cb = std::move(on_output)]() mutable {
        reader_thread_fn(std::move(cb));
    });

    return true;
}

void subprocess_handle::reader_thread_fn(output_cb on_output) {
    FILE * fp = subprocess_stdout(&m_proc); // combined stdout+stderr
    if (!fp) {
        return;
    }

    char line_buf[4096];
    while (!m_stop_requested.load()) {
        if (fgets(line_buf, sizeof(line_buf), fp) == nullptr) {
            break; // EOF or error
        }
        // Strip trailing newline
        size_t len = strlen(line_buf);
        while (len > 0 && (line_buf[len - 1] == '\n' || line_buf[len - 1] == '\r')) {
            line_buf[--len] = '\0';
        }
        if (on_output) {
            on_output(std::string(line_buf, len));
        }
    }
}

bool subprocess_handle::write_stdin(const std::string & msg) {
    if (!m_spawned.load() || !is_alive()) {
        return false;
    }
    FILE * fp = subprocess_stdin(&m_proc);
    if (!fp) {
        return false;
    }
    if (fprintf(fp, "%s\n", msg.c_str()) < 0) {
        return false;
    }
    fflush(fp);
    return true;
}

bool subprocess_handle::is_alive() const {
    if (!m_spawned.load()) {
        return false;
    }
    return subprocess_alive(const_cast<subprocess_s *>(&m_proc)) != 0;
}

void subprocess_handle::stop(int timeout_ms) {
    if (!m_spawned.load()) {
        return;
    }

    m_stop_requested.store(true);

    if (is_alive()) {
        // Close stdin to signal EOF to child
        FILE * fp = subprocess_stdin(&m_proc);
        if (fp) {
            fclose(fp);
        }

#ifndef _WIN32
        // Send SIGTERM for graceful shutdown
        kill(m_proc.child, SIGTERM);
#endif

        // Wait up to timeout_ms for the process to exit
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(timeout_ms);

        while (is_alive() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (is_alive()) {
            LOG_WRN("subprocess_handle: process did not exit within timeout, terminating\n");
            subprocess_terminate(&m_proc);
        }
    }

    // Join the reader thread
    if (m_reader.joinable()) {
        m_reader.join();
    }

    // Collect exit code and release OS resources
    int code = 0;
    subprocess_join(&m_proc, &code);
    subprocess_destroy(&m_proc);

    m_exit_code.store(code);
    m_spawned.store(false);
}

int subprocess_handle::exit_code() const {
    return m_exit_code.load();
}
