#pragma once

#include <sheredom/subprocess.h>

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>

/**
 * RAII wrapper managing a single child process for a tool executor.
 *
 * Lifecycle:
 *   1. spawn(argv, env)    — start the process
 *   2. write_stdin(msg)    — send a line to the child's stdin
 *   3. is_alive()          — poll liveness
 *   4. stop(timeout_ms)   — graceful shutdown, then force-kill
 *   5. destructor          — guarantees cleanup
 *
 * stdout/stderr is read by a background reader thread whose output is
 * forwarded to the on_output callback.
 */
class subprocess_handle {
public:
    using output_cb = std::function<void(const std::string & line)>;

    subprocess_handle();
    ~subprocess_handle();

    // Non-copyable, non-movable (thread holds 'this')
    subprocess_handle(const subprocess_handle &)            = delete;
    subprocess_handle & operator=(const subprocess_handle &) = delete;
    subprocess_handle(subprocess_handle &&)                  = delete;
    subprocess_handle & operator=(subprocess_handle &&)      = delete;

    /**
     * Start a child process.
     *
     * argv:      argument strings (argv[0] = program path)
     * env:       "KEY=VALUE" strings (empty → inherit parent environment)
     * on_output: callback invoked for each line from the child's stdout/stderr
     *
     * Returns true on success.
     */
    bool spawn(const std::vector<std::string> & argv,
               const std::vector<std::string> & env,
               output_cb on_output = nullptr);

    /** Send a line to the child's stdin (appends '\n'). */
    bool write_stdin(const std::string & msg);

    /** True while the child process is running. */
    bool is_alive() const;

    /**
     * Graceful stop: write EOF / signal child, wait up to timeout_ms,
     * then force-terminate.  Joins the reader thread.
     */
    void stop(int timeout_ms = 5000);

    /** Exit code after the process has finished (-1 if not yet finished). */
    int exit_code() const;

private:
    void reader_thread_fn(output_cb on_output);

    subprocess_s          m_proc{};            // in-place; matches server-models.cpp pattern
    std::thread           m_reader;
    std::atomic<bool>     m_stop_requested{false};
    std::atomic<int>      m_exit_code{-1};
    std::atomic<bool>     m_spawned{false};
};
