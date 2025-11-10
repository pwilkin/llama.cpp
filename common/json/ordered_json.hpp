#pragma once

// Ordered JSON value representation
// This header defines a set of lightweight data structures that model JSON values
// while preserving the insertion order of object members. All types are implemented
// using only the C++ standard library.

#include <cassert>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <map>
#include <cstdio>

// Helper for std::visit overload set.
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...)->overloaded<Ts...>;

// Forward declaration
struct Node;

/* -------------------------------------------------------------------------- */
/*  Primitive                                                               */
/* -------------------------------------------------------------------------- */
// JSON primitive values: null, boolean, number, string.
using Primitive = std::variant<std::nullptr_t, bool, double, std::string>;

/* -------------------------------------------------------------------------- */
/*  BaseValue                                                               */
/* -------------------------------------------------------------------------- */
// Abstract base class for polymorphic storage of concrete JSON values.
struct BaseValue {
    virtual ~BaseValue() = default;
};

/* -------------------------------------------------------------------------- */
/*  OrderedObject                                                          */
/* -------------------------------------------------------------------------- */
// Object that preserves the order in which keys are inserted.
struct OrderedObject {
    // Internal storage: vector of (key, value) pairs.
    std::vector<std::pair<std::string, Node>> entries_;

    // Insert a new key/value pair. If the key already exists, its value is
    // replaced without changing the order.
    void insert(const std::string& key, Node value) {
        for (auto& kv : entries_) {
            if (kv.first == key) {
                kv.second = std::move(value);
                return;
            }
        }
        entries_.emplace_back(key, std::move(value));
    }

    // Find a value by key. Returns a pointer to the value or nullptr if missing.
    Node* find(const std::string& key) {
        for (auto& kv : entries_) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    // Const overload of find.
    const Node* find(const std::string& key) const {
        for (const auto& kv : entries_) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    // Access the ordered list of entries.
    const std::vector<std::pair<std::string, Node>>& entries() const {
        return entries_;
    }

    // Return a map view. The map is populated in insertion order, but iteration
    // follows the map's key ordering (standard std::map). This provides a quick
    // lookup by key while still allowing iteration that respects insertion order
    // if the caller iterates over the original entries vector.
    std::map<std::string, Node> to_ordered_map() const {
        std::map<std::string, Node> m;
        for (const auto& kv : entries_) {
            m.emplace(kv.first, kv.second);
        }
        return m;
    }

    // Equality comparison (used by Node::operator==).
    bool operator==(const OrderedObject& other) const {
        return entries_ == other.entries_;
    }
};

/* -------------------------------------------------------------------------- */
/*  Array                                                                   */
/* -------------------------------------------------------------------------- */
// JSON array is a simple vector of Nodes.
using Array = std::vector<Node>;

/* -------------------------------------------------------------------------- */
/*  Concrete value wrappers                                                  */
/* -------------------------------------------------------------------------- */
struct ObjectValue : BaseValue {
    OrderedObject obj;
    explicit ObjectValue(OrderedObject o) : obj(std::move(o)) {}
    bool operator==(const ObjectValue& other) const { return obj == other.obj; }
};

struct ArrayValue : BaseValue {
    Array arr;
    explicit ArrayValue(Array a) : arr(std::move(a)) {}
    bool operator==(const ArrayValue& other) const { return arr == other.arr; }
};

struct PrimitiveValue : BaseValue {
    Primitive prim;
    explicit PrimitiveValue(const Primitive& p) : prim(p) {}
    explicit PrimitiveValue(Primitive&& p) : prim(std::move(p)) {}
    bool operator==(const PrimitiveValue& other) const { return prim == other.prim; }
};

/* -------------------------------------------------------------------------- */
/*  Node                                                                    */
/* -------------------------------------------------------------------------- */
enum class NodeType { Object, Array, Primitive, Custom };

struct Node {
    NodeType type_;
    std::unique_ptr<BaseValue> value_;

    // Default constructor creates a null primitive.
    Node() : type_(NodeType::Primitive), value_(std::make_unique<PrimitiveValue>(nullptr)) {}

    // Constructors for each concrete type.
    explicit Node(const OrderedObject& obj)
        : type_(NodeType::Object), value_(std::make_unique<ObjectValue>(obj)) {}

    explicit Node(OrderedObject&& obj)
        : type_(NodeType::Object), value_(std::make_unique<ObjectValue>(std::move(obj))) {}

    explicit Node(const Array& arr)
        : type_(NodeType::Array), value_(std::make_unique<ArrayValue>(arr)) {}

    explicit Node(Array&& arr)
        : type_(NodeType::Array), value_(std::make_unique<ArrayValue>(std::move(arr))) {}

    explicit Node(const Primitive& prim)
        : type_(NodeType::Primitive), value_(std::make_unique<PrimitiveValue>(prim)) {}

    explicit Node(Primitive&& prim)
        : type_(NodeType::Primitive), value_(std::make_unique<PrimitiveValue>(std::move(prim))) {}

    // Accessors – they assert that the stored type matches the requested view.
    OrderedObject& as_object() {
        assert(type_ == NodeType::Object);
        return static_cast<ObjectValue*>(value_.get())->obj;
    }
    const OrderedObject& as_object() const {
        assert(type_ == NodeType::Object);
        return static_cast<const ObjectValue*>(value_.get())->obj;
    }

    Array& as_array() {
        assert(type_ == NodeType::Array);
        return static_cast<ArrayValue*>(value_.get())->arr;
    }
    const Array& as_array() const {
        assert(type_ == NodeType::Array);
        return static_cast<const ArrayValue*>(value_.get())->arr;
    }

    Primitive& as_primitive() {
        assert(type_ == NodeType::Primitive);
        return static_cast<PrimitiveValue*>(value_.get())->prim;
    }
    const Primitive& as_primitive() const {
        assert(type_ == NodeType::Primitive);
        return static_cast<const PrimitiveValue*>(value_.get())->prim;
    }

    // Equality operator for testing.
    bool operator==(const Node& other) const {
        if (type_ != other.type_) return false;
        switch (type_) {
            case NodeType::Object:
                return *static_cast<const ObjectValue*>(value_.get()) ==
                       *static_cast<const ObjectValue*>(other.value_.get());
            case NodeType::Array:
                return *static_cast<const ArrayValue*>(value_.get()) ==
                       *static_cast<const ArrayValue*>(other.value_.get());
            case NodeType::Primitive:
                return *static_cast<const PrimitiveValue*>(value_.get()) ==
                       *static_cast<const PrimitiveValue*>(other.value_.get());
            case NodeType::Custom:
                // Custom type comparison is user‑defined; fallback to address equality.
                return value_.get() == other.value_.get();
        }
        return false; // Unreachable.
    }
};

/* -------------------------------------------------------------------------- */
/*  Utility functions                                                       */
/* -------------------------------------------------------------------------- */
inline bool is_ordered_object(const Node& n) { return n.type_ == NodeType::Object; }
inline bool is_array(const Node& n)           { return n.type_ == NodeType::Array; }
inline bool is_primitive(const Node& n)      { return n.type_ == NodeType::Primitive; }

inline std::vector<Node> to_vector(const Node& n) {
    assert(is_array(n));
    return n.as_array();
}

inline std::map<std::string, Node> to_ordered_map(const Node& n) {
    assert(is_ordered_object(n));
    return n.as_object().to_ordered_map();
}
// Serialize a Node to a JSON string preserving key order.
// This utility is used by the test suite.
inline std::string json_dump(const Node& n) {
    // Helper to escape strings.
    auto escape = [](const std::string& s) {
        std::string out = "\"";
        for (char c : s) {
            switch (c) {
                case '\"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out += c;
                    }
            }
        }
        out += "\"";
        return out;
    };

    if (is_primitive(n)) {
        const Primitive& p = n.as_primitive();
        return std::visit(overloaded{
            [](std::nullptr_t) { return std::string("null"); },
            [](bool b) { return b ? std::string("true") : std::string("false"); },
            [](double d) {
                // Use default formatting; for integers this will include ".0" which is acceptable for tests.
                return std::to_string(d);
            },
            [&](const std::string& s) { return escape(s); }
        }, p);
    } else if (is_array(n)) {
        const Array& arr = n.as_array();
        std::string out = "[";
        bool first = true;
        for (const auto& elem : arr) {
            if (!first) out += ",";
            out += json_dump(elem);
            first = false;
        }
        out += "]";
        return out;
    } else if (is_ordered_object(n)) {
        const OrderedObject& obj = n.as_object();
        std::string out = "{";
        bool first = true;
        for (const auto& kv : obj.entries()) {
            if (!first) out += ",";
            out += escape(kv.first);
            out += ":";
            out += json_dump(kv.second);
            first = false;
        }
        out += "}";
        return out;
    } else {
        // Custom types not supported in this simple serializer.
        return std::string("<custom>");
    }
}