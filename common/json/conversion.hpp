#pragma once

// Conversion utilities for the ordered JSON model defined in ordered_json.hpp.
// These functions provide a simple, header‑only interface to transform the internal
// representation (Array, OrderedObject, Primitive) into standard C++ containers.
//
// The implementation relies only on the C++ standard library and the definitions
// from "ordered_json.hpp". All functions are inline to avoid linker issues.

#include "ordered_json.hpp"
#include <cassert>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

/**
 * @brief Alias for an ordered map view.
 *
 * The alias uses std::map with the default comparator. Insertion is performed in
 * the order of the original JSON object, so the map contains the same key/value
 * pairs. Iteration order of std::map is sorted by key; however, because the map
 * is populated following the insertion order, callers that need to preserve that
 * order can rely on the underlying OrderedObject::entries() vector instead.
 *
 * If strict insertion‑order iteration is required, consider using the
 * `OrderedObject::entries()` directly.
 */
using OrderedMap = std::map<std::string, Node>;

/**
 * @brief Convert an Array (std::vector<Node>) to a std::vector<Node>.
 *
 * This is essentially an identity conversion, provided for API symmetry.
 *
 * @param arr The ordered JSON array.
 * @return A copy of the array as a std::vector<Node>.
 */
inline std::vector<Node> to_vector(const Array& arr) {
    return arr; // copy – callers receive their own vector.
}

/**
 * @brief Convert an OrderedObject to an OrderedMap.
 *
 * The map is populated by iterating over the object's entries in insertion order.
 * The resulting map can be used for fast lookup by key.
 *
 * @param obj The ordered JSON object.
 * @return An OrderedMap containing the same key/value pairs.
 */
inline OrderedMap to_ordered_map(const OrderedObject& obj) {
    OrderedMap result;
    for (const auto& kv : obj.entries()) {
        result.emplace(kv.first, kv.second);
    }
    return result;
}

/**
 * @brief Extract a primitive value from a Primitive variant.
 *
 * Supported target types are:
 *   - bool
 *   - double
 *   - std::string
 *
 * The function throws std::bad_variant_access if the stored type does not match
 * the requested type.
 *
 * @tparam T The desired primitive type.
 * @param p The Primitive variant.
 * @return The extracted value of type T.
 */
template <typename T>
inline T to_primitive(const Primitive& p) {
    static_assert(std::is_same_v<T, bool> ||
                  std::is_same_v<T, double> ||
                  std::is_same_v<T, std::string>,
                  "Unsupported primitive type requested");

    if constexpr (std::is_same_v<T, bool>) {
        return std::get<bool>(p);
    } else if constexpr (std::is_same_v<T, double>) {
        return std::get<double>(p);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return std::get<std::string>(p);
    } else {
        // This point should never be reached due to static_assert.
        throw std::bad_variant_access{};
    }
}

inline std::vector<Node> to_vector(const Node& n) {
    assert(is_array(n) && "Node does not contain an Array");
    if (!is_array(n)) {
        throw std::logic_error("Node does not contain an Array");
    }
    return n.as_array();
}

/**
 * @brief Convert a Node containing an OrderedObject to an OrderedMap.
 *
 * @param n Node that must hold an Object.
 * @return The extracted OrderedMap.
 * @throws std::logic_error if the Node does not contain an Object.
 */
inline OrderedMap to_ordered_map(const Node& n) {
    assert(is_ordered_object(n) && "Node does not contain an Object");
    if (!is_ordered_object(n)) {
        throw std::logic_error("Node does not contain an Object");
    }
    return n.as_object().to_ordered_map();
}


/**
 * @brief Extract a primitive value from a Node.
 *
 * The function forwards to to_primitive<T> after confirming the Node holds a
 * Primitive.
 *
 * @tparam T Desired primitive type (bool, double, std::string).
 * @param n Node that must hold a Primitive.
 * @return The extracted primitive value.
 * @throws std::logic_error if the Node does not contain a Primitive.
 */
template <typename T>
inline T to_primitive(const Node& n) {
    assert(is_primitive(n) && "Node does not contain a Primitive");
    if (!is_primitive(n)) {
        throw std::logic_error("Node does not contain a Primitive");
    }
    return to_primitive<T>(n.as_primitive());
}