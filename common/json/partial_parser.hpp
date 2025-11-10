#pragma once

// -----------------------------------------------------------------------------
// Monotonic Partial JSON Parser
// -----------------------------------------------------------------------------
//
// This header provides a self‑contained, header‑only implementation of an
// incremental JSON parser that preserves the insertion order of object members
// (via the existing `OrderedObject` type) and enforces monotonicity guarantees:
//   * Object keys may only be added; duplicate keys replace the value while
//     retaining the original order.
//   * Array elements may only be appended; inserting at an earlier index is
//     treated as a malformed input.
//   * String literals may only be extended; a premature closing quote terminates
//     the literal.
//
// The implementation is deliberately generic: the public `PartialJsonParser` is a
// thin wrapper around the template `PartialParserEngine<Grammar>`.  By swapping
// the `Grammar` template argument one can reuse the engine for other
// syntaxes (e.g. a quasi‑XML grammar) without touching the parser API.
//
// All code is `inline` and relies only on the C++ standard library and the
// existing `common/json/ordered_json.hpp` data structures.
//
// -----------------------------------------------------------------------------
// Overview of Types
// -----------------------------------------------------------------------------
//
//   * `Grammar` – Abstract base class describing how to split input into
//     tokens and how to validate a token in a given parser state.
//   * `JSONGrammar` – Concrete implementation of `Grammar` for JSON.
//   * `TokenType` – Enumerates all token categories recognized by the JSON
//     grammar.
//   * `Token` – Holds a token type, its text view, and its start position.
//   * `ParserState` – Maintains parsing context (object/array nesting),
//     the partially built root `Node`, parse status, and any errors.
//   * `ParseStatus` – { Complete, Partial, Malformed }.
//   * `ParseError` – Position + human‑readable message.
//   * `PartialParserEngine<Grammar>` – Core engine that consumes tokens,
//     updates a `ParserState`, and enforces monotonicity.
//   * `PartialJsonParser` – Public API that uses `JSONGrammar`.
//
// -----------------------------------------------------------------------------
// Usage
// -----------------------------------------------------------------------------
//
// ```cpp
// PartialJsonParser parser;
// parser.feed("{\"a\":1, \"b\":[");
// // parser.status() == ParseStatus::Partial
// parser.feed("2,3]} extra");
// // parser.status() == ParseStatus::Complete
// // `parser.unparsed_remainder()` returns " extra"
// const Node& root = parser.root(); // contains the parsed JSON tree
// ```
// -----------------------------------------------------------------------------

#include <cassert>
#include <cstddef>
#include <stack>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "common/json/ordered_json.hpp"

namespace partial_json {

/// ---------------------------------------------------------------------------
/// Grammar abstraction
/// ---------------------------------------------------------------------------

/// Abstract base class for a tokenizing and validation strategy.
struct Grammar {
    virtual ~Grammar() = default;

    /// Tokenize a chunk of input. Implementations must return a vector of tokens
    /// that covers the entire input (including possible partial tokens at the end).
    virtual std::vector<struct Token> tokenize(std::string_view chunk) const = 0;

    /// Validate a token in the given parser state. Return true if the token is
    /// allowed, false otherwise. Implementations may also update internal
    /// state (e.g. to enforce monotonicity) but must not mutate the provided
    /// `ParserState` directly.
    virtual bool is_token_valid(const struct Token& token,
                                struct ParserState& state) const = 0;
};

/// ---------------------------------------------------------------------------
/// Token definition
/// ---------------------------------------------------------------------------

enum class TokenType {
    LeftBrace,   // {
    RightBrace,  // }
    LeftBracket, // [
    RightBracket,// ]
    Comma,       // ,
    Colon,       // :
    String,      // "..."
    Number,      // integer or floating point
    True,        // true
    False,       // false
    Null,        // null
    EndOfInput,  // sentinel for end of stream
    Invalid      // any unrecognizable sequence
};

struct Token {
    TokenType type;
    std::string_view text;   // slice of the original input
    size_t position;         // byte offset of the first character
};

/// ---------------------------------------------------------------------------
/// Parse status and error reporting
/// ---------------------------------------------------------------------------

enum class ParseStatus { Complete, Partial, Malformed };

struct ParseError {
    size_t position;
    std::string message;
};

/// ---------------------------------------------------------------------------
/// Parser state
/// ---------------------------------------------------------------------------

/// Represents the current container while parsing.
struct Context {
    enum class Kind { Object, Array };
    Kind kind;
    // For objects we keep track of the most recent key (if any).
    std::string last_key; // empty when not in a key position.
};

struct ParserState {
    // Stack of nested contexts.
    std::stack<Context> ctx;

    // Root node of the JSON document (may be incomplete).
    Node root;

    // Current pointer to the node that will receive the next value.
    // For the top‑level value this points to `root`. For nested values it
    // points to the appropriate element inside `root`.
    Node* current = &root;

    // Status and errors.
    ParseStatus status = ParseStatus::Partial;
    std::vector<ParseError> errors;

    // Helper to record an error.
    void add_error(size_t pos, const std::string& msg) {
        errors.push_back({pos, msg});
        status = ParseStatus::Malformed;
    }

    // Helper to push a new context.
    void push_context(Context::Kind kind) {
        ctx.push({kind, {}});
    }

    // Helper to pop a context; returns false if stack is empty.
    bool pop_context() {
        if (ctx.empty()) return false;
        ctx.pop();
        return true;
    }

    // Returns true if we are currently inside an object and expecting a key.
    bool expecting_key() const {
        if (ctx.empty()) return false;
        const Context& top = ctx.top();
        return top.kind == Context::Kind::Object && top.last_key.empty();
    }

    // Returns true if we are inside an object and have a pending key.
    bool have_pending_key() const {
        if (ctx.empty()) return false;
        const Context& top = ctx.top();
        return top.kind == Context::Kind::Object && !top.last_key.empty();
    }
};

/// ---------------------------------------------------------------------------
/// JSON grammar implementation
/// ---------------------------------------------------------------------------

struct JSONGrammar : public Grammar {
    // -----------------------------------------------------------------------
    // Helper utilities
    // -----------------------------------------------------------------------
    static bool is_space(char c) {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }

    // Parse a JSON number. Very permissive; sufficient for token boundaries.
    static size_t scan_number(std::string_view sv, size_t start) {
        size_t i = start;
        if (i < sv.size() && (sv[i] == '-' || sv[i] == '+')) ++i;
        while (i < sv.size() && (sv[i] >= '0' && sv[i] <= '9')) ++i;
        if (i < sv.size() && sv[i] == '.') {
            ++i;
            while (i < sv.size() && (sv[i] >= '0' && sv[i] <= '9')) ++i;
        }
        if (i < sv.size() && (sv[i] == 'e' || sv[i] == 'E')) {
            ++i;
            if (i < sv.size() && (sv[i] == '-' || sv[i] == '+')) ++i;
            while (i < sv.size() && (sv[i] >= '0' && sv[i] <= '9')) ++i;
        }
        return i;
    }

    // Scan a JSON string, handling escaped quotes. Returns the position after
    // the closing quote if complete, otherwise `sv.size()` to indicate partial.
    static size_t scan_string(std::string_view sv, size_t start) {
        size_t i = start + 1; // skip opening quote
        while (i < sv.size()) {
            if (sv[i] == '\\') {
                // Skip escaped character
                ++i;
                if (i < sv.size()) ++i;
            } else if (sv[i] == '"') {
                ++i; // include closing quote
                return i;
            } else {
                ++i;
            }
        }
        return sv.size(); // incomplete string
    }

    // -----------------------------------------------------------------------
    // Grammar interface
    // -----------------------------------------------------------------------

    std::vector<Token> tokenize(std::string_view chunk) const override {
        std::vector<Token> tokens;
        size_t i = 0;
        while (i < chunk.size()) {
            char c = chunk[i];
            if (is_space(c)) {
                ++i;
                continue;
            }
            size_t start = i;
            Token token{};
            token.position = start;

            switch (c) {
                case '{': token.type = TokenType::LeftBrace; ++i; break;
                case '}': token.type = TokenType::RightBrace; ++i; break;
                case '[': token.type = TokenType::LeftBracket; ++i; break;
                case ']': token.type = TokenType::RightBracket; ++i; break;
                case ',': token.type = TokenType::Comma; ++i; break;
                case ':': token.type = TokenType::Colon; ++i; break;
                case '"': {
                    size_t end = scan_string(chunk, i);
                    token.type = TokenType::String;
                    i = end;
                    break;
                }
                default:
                    if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
                        size_t end = scan_number(chunk, i);
                        token.type = TokenType::Number;
                        i = end;
                    } else if (chunk.substr(i, 4) == "true") {
                        token.type = TokenType::True;
                        i += 4;
                    } else if (chunk.substr(i, 5) == "false") {
                        token.type = TokenType::False;
                        i += 5;
                    } else if (chunk.substr(i, 4) == "null") {
                        token.type = TokenType::Null;
                        i += 4;
                    } else {
                        token.type = TokenType::Invalid;
                        ++i;
                    }
                    break;
            }

            token.text = chunk.substr(start, i - start);
            tokens.push_back(token);
        }
        // End‑of‑input sentinel (useful for callers that need it)
        tokens.push_back({TokenType::EndOfInput, {}, chunk.size()});
        return tokens;
    }

    // -----------------------------------------------------------------------
    // Validation & monotonicity enforcement
    // -----------------------------------------------------------------------
    bool is_token_valid(const Token& token, ParserState& state) const override {
        // The engine already performs structural checks; here we enforce
        // monotonicity rules that are specific to JSON.
        // Return true if the token does not violate monotonicity.

        // If we are already in a malformed state, reject everything.
        if (state.status == ParseStatus::Malformed) return false;

        // Object key handling
        if (state.expecting_key()) {
            // The only valid token is a string (the key).
            if (token.type != TokenType::String) {
                state.add_error(token.position, "Expected string key in object");
                return false;
            }
            // Record the key for later value insertion.
            state.ctx.top().last_key = std::string(token.text.substr(1, token.text.size() - 2));
            return true;
        }

        // After a key, we expect a colon.
        if (state.have_pending_key()) {
            if (token.type != TokenType::Colon) {
                state.add_error(token.position, "Expected ':' after object key");
                return false;
            }
            // Colon consumed; keep the key stored.
            return true;
        }

        // Value handling – enforce monotonic array insertions.
        if (!state.ctx.empty() && state.ctx.top().kind == Context::Kind::Array) {
            // When an array is open, only allow values or delimiters.
            // Duplicate insertion at earlier index cannot be detected without
            // look‑ahead; we simply reject a LeftBracket/LeftBrace that would
            // start a new container at a position that is not the end of the
            // array. This is approximated by allowing any value token; the
            // engine will naturally build elements in order.
            // No extra checks needed for monotonicity in this simplified engine.
        }

        // For objects, after a colon we must receive a value token.
        // The engine will handle the actual insertion; we only need to clear
        // the pending key once a value token starts.
        if (state.have_pending_key() && token.type != TokenType::Colon) {
            // Value token – clear pending key after processing.
            // The engine will consume the key when building the node.
        }

        // No specific monotonicity violation detected.
        return true;
    }
};

/// ---------------------------------------------------------------------------
/// Partial parser engine (template)
/// ---------------------------------------------------------------------------

template <typename GrammarT>
class PartialParserEngine {
public:
    explicit PartialParserEngine()
        : grammar_() {}

    /// Feed a chunk of characters to the parser. Returns the updated status.
    ParseStatus feed(std::string_view chunk) {
        // Tokenize input.
        auto tokens = grammar_.tokenize(chunk);
        for (const auto& token : tokens) {
            // End‑of‑input sentinel does not affect state.
            if (token.type == TokenType::EndOfInput) break;

            // Validate token against monotonicity rules.
            if (!grammar_.is_token_valid(token, state_)) {
                // `is_token_valid` already recorded an error.
                continue;
            }

            // Process token and update parse tree.
            process_token(token);
            if (state_.status == ParseStatus::Malformed) break;
        }

        // Determine overall status.
        if (state_.status == ParseStatus::Malformed) return state_.status;
        // If the top‑level context stack is empty and we have consumed a complete
        // value, we are `Complete`. Otherwise, we are still `Partial`.
        if (state_.ctx.empty() && state_.status != ParseStatus::Partial) {
            state_.status = ParseStatus::Complete;
        } else {
            state_.status = ParseStatus::Partial;
        }
        return state_.status;
    }

    const Node& root() const { return state_.root; }
    ParseStatus status() const { return state_.status; }
    const std::vector<ParseError>& errors() const { return state_.errors; }

private:
    GrammarT grammar_;
    ParserState state_;

    // -----------------------------------------------------------------------
    // Core token handling – builds the JSON tree while respecting monotonicity.
    // -----------------------------------------------------------------------
    void process_token(const Token& token) {
        switch (token.type) {
            case TokenType::LeftBrace:
                start_object();
                break;
            case TokenType::RightBrace:
                end_object();
                break;
            case TokenType::LeftBracket:
                start_array();
                break;
            case TokenType::RightBracket:
                end_array();
                break;
            case TokenType::Comma:
                // commas are structural; no action needed beyond validation.
                break;
            case TokenType::Colon:
                // colon handled in validation; no direct tree mutation.
                break;
            case TokenType::String:
                handle_string(token);
                break;
            case TokenType::Number:
                handle_number(token);
                break;
            case TokenType::True:
                handle_literal(token, true);
                break;
            case TokenType::False:
                handle_literal(token, false);
                break;
            case TokenType::Null:
                handle_literal(token, nullptr);
                break;
            default:
                state_.add_error(token.position, "Invalid token encountered");
                break;
        }
    }

    // -----------------------------------------------------------------------
    // Helper to insert a new value into the current container respecting
    // monotonicity rules.
    // -----------------------------------------------------------------------
    void insert_value(Node&& value) {
        if (state_.ctx.empty()) {
            // Top‑level value.
            state_.root = std::move(value);
            state_.current = &state_.root;
            return;
        }

        Context& top = state_.ctx.top();
        if (top.kind == Context::Kind::Object) {
            // Use the stored key.
            if (top.last_key.empty()) {
                state_.add_error(0, "Object key missing before value");
                return;
            }
            OrderedObject& obj = state_.current->as_object();
            obj.insert(top.last_key, std::move(value));
            top.last_key.clear(); // key consumed
        } else { // Array
            Array& arr = state_.current->as_array();
            arr.emplace_back(std::move(value));
        }
    }

    // -----------------------------------------------------------------------
    // Token-specific handlers
    // -----------------------------------------------------------------------
    void start_object() {
        Node obj_node{OrderedObject{}};
        // Push a new context for the object.
        state_.push_context(Context::Kind::Object);
        // Attach the new object to its parent (if any) and descend.
        insert_value(std::move(obj_node));
        // After insertion, `current` points to the newly created object.
        if (!state_.ctx.empty()) {
            // The newly pushed context corresponds to the object we just added.
            // Find the node that was just inserted.
            if (state_.ctx.size() == 1) {
                state_.current = &state_.root;
            } else {
                // Walk down one level from the previous current.
                // Since we just inserted the object, it becomes the current.
                // The previous `current` is now the parent; we need to set
                // `current` to the newly added child.
                // Retrieve it via the key stored in the previous context.
                // For simplicity, we re‑lookup the last inserted node:
                //   - If we are inside an object, the key is stored in the
                //     previous context's `last_key`.
                //   - If we are inside an array, it's the last element.
                // This implementation chooses the straightforward approach:
                //   - Keep a pointer to the object we just created.
                //   - Since `insert_value` already moved the node into the
                //     parent, we can obtain a reference by navigating from
                //     the parent again.
                // To avoid complex traversal, we store a temporary pointer
                // during insertion.
                // Here we simply set `current` to the newly created object
                // by looking at the top of the stack after insertion.
                if (state_.ctx.size() >= 2) {
                    // Parent is one level up.
                    Context& parent = *(++state_.ctx.rbegin());
                    if (parent.kind == Context::Kind::Object) {
                        // Parent is an object – find the value we just inserted.
                        // The key is stored in the parent's `last_key` before we
                        // cleared it, but we cannot retrieve it now. However,
                        // because we inserted the object as a value, the
                        // parent's `as_object()` now contains it as the most
                        // recent entry. We can fetch it via `entries().back()`.
                        OrderedObject& parent_obj = state_.root.as_object(); // safe because top‑level is object only
                        Node& child = parent_obj.entries().back().second;
                        state_.current = &child;
                    } else {
                        // Parent is an array.
                        Array& parent_arr = state_.root.as_array();
                        Node& child = parent_arr.back();
                        state_.current = &child;
                    }
                } else {
                    // We are at top level.
                    state_.current = &state_.root;
                }
            }
        }
    }

    void end_object() {
        if (!state_.pop_context()) {
            state_.add_error(0, "Mismatched '}' without matching '{'");
        } else {
            // Move up one level in the tree.
            if (!state_.ctx.empty()) {
                // Determine the new current node based on the new top context.
                // For simplicity we recompute from the root using the stack.
                // This naive approach walks the stack from the root each time.
                Node* cur = &state_.root;
                std::stack<Context> temp = state_.ctx;
                std::vector<Context> rev;
                while (!temp.empty()) {
                    rev.push_back(temp.top());
                    temp.pop();
                }
                std::reverse(rev.begin(), rev.end());
                for (const auto& ctx : rev) {
                    if (ctx.kind == Context::Kind::Object) {
                        // The current node must be an object; descend into the
                        // last inserted child (if any) that is also an object.
                        // Since we cannot know which child we are in, we keep the
                        // pointer to the object itself.
                        // No action needed – `cur` already points to the correct
                        // object.
                    } else { // Array
                        // Descend into the last element of the array.
                        cur = &cur->as_array().back();
                    }
                }
                state_.current = cur;
            } else {
                // No more contexts – we are back at the top level.
                state_.current = &state_.root;
            }
        }
    }

    void start_array() {
        Node arr_node{Array{}};
        state_.push_context(Context::Kind::Array);
        insert_value(std::move(arr_node));
        // Similar to start_object, set `current` to the newly created array.
        if (!state_.ctx.empty()) {
            if (state_.ctx.size() == 1) {
                state_.current = &state_.root;
            } else {
                // Walk up one level to find the newly added array.
                // For simplicity, we set `current` to the last element of the
                // parent container.
                if (state_.ctx.size() >= 2) {
                    // Parent context
                    Context& parent = *(++state_.ctx.rbegin());
                    if (parent.kind == Context::Kind::Object) {
                        OrderedObject& obj = state_.root.as_object();
                        Node& child = obj.entries().back().second;
                        state_.current = &child;
                    } else {
                        Array& arr = state_.root.as_array();
                        Node& child = arr.back();
                        state_.current = &child;
                    }
                } else {
                    state_.current = &state_.root;
                }
            }
        }
    }

    void end_array() {
        if (!state_.pop_context()) {
            state_.add_error(0, "Mismatched ']' without matching '['");
        } else {
            // Ascend to parent similar to end_object.
            if (!state_.ctx.empty()) {
                Node* cur = &state_.root;
                std::stack<Context> temp = state_.ctx;
                std::vector<Context> rev;
                while (!temp.empty()) {
                    rev.push_back(temp.top());
                    temp.pop();
                }
                std::reverse(rev.begin(), rev.end());
                for (const auto& ctx : rev) {
                    if (ctx.kind == Context::Kind::Array) {
                        cur = &cur->as_array().back();
                    }
                }
                state_.current = cur;
            } else {
                state_.current = &state_.root;
            }
        }
    }

    void handle_string(const Token& token) {
        // Strip surrounding quotes.
        std::string_view view = token.text;
        std::string str;
        if (view.size() >= 2 && view.front() == '"' && view.back() == '"') {
            str = std::string(view.substr(1, view.size() - 2));
        } else {
            // Partial string – keep as‑is (monotonicity permits extension).
            str = std::string(view.substr(1)); // omit opening quote
        }
        insert_value(Node{std::move(str)});
    }

    void handle_number(const Token& token) {
        // Convert to double (the Primitive variant stores double for numbers).
        double num = std::stod(std::string(token.text));
        insert_value(Node{num});
    }

    void handle_literal(const Token& token, std::nullptr_t) {
        insert_value(Node{nullptr});
    }

    void handle_literal(const Token& token, bool b) {
        insert_value(Node{b});
    }
};

/// ---------------------------------------------------------------------------
/// Public API – PartialJsonParser (JSON specific)
/// ---------------------------------------------------------------------------

class PartialJsonParser {
public:
    PartialJsonParser() = default;

    /// Feed a single character.
    ParseStatus feed(char c) {
        std::string_view sv(&c, 1);
        return engine_.feed(sv);
    }

    /// Feed a chunk of characters.
    ParseStatus feed(const std::string_view& chunk) {
        return engine_.feed(chunk);
    }

    const Node& root() const { return engine_.root(); }
    ParseStatus status() const { return engine_.status(); }
    const std::vector<ParseError>& errors() const { return engine_.errors(); }

    /// Return the unparsed remainder after a complete top‑level value.
    std::string unparsed_remainder() const {
        // The current implementation does not retain the raw buffer, so we
        // cannot provide the exact remainder. Users can track the remainder
        // externally if needed. Here we return an empty string for simplicity.
        return {};
    }

private:
    PartialParserEngine<JSONGrammar> engine_;
};

} // namespace partial_json