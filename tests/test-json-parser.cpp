#include "common/json/ordered_json.hpp"
#include "common/json/conversion.hpp"
#include "common/json/partial_parser.hpp"
#include <cassert>
#include <iostream>

using partial_json::PartialJsonParser;
using partial_json::ParseStatus;

void expect_status(const PartialJsonParser& p, ParseStatus expected) {
    assert(p.status() == expected);
}

void test_complete_object() {
    PartialJsonParser p;
    std::string json = R"({"b":2,"a":1})";
    p.feed(std::string_view(json));
    expect_status(p, ParseStatus::Complete);
    std::string dump = json_dump(p.root());
    assert(dump == R"({"b":2,"a":1})");
}

void test_key_order_preservation() {
    PartialJsonParser p;
    std::string json = R"({"first":1,"asecond":2,"bthird":3})";
    p.feed(std::string_view(json));
    expect_status(p, ParseStatus::Complete);
    std::string dump = json_dump(p.root());
    assert(dump == R"({"first":1,"asecond":2,"bthird":3})");
}

void test_incremental_parsing() {
    PartialJsonParser p;
    std::string json = R"([1,2,3])";
    for (char c : json) {
        p.feed(std::string_view(&c,1));
        if (&c != &json.back()) {
            assert(p.status() == ParseStatus::Partial);
        }
    }
    expect_status(p, ParseStatus::Complete);
}

void test_malformed_input() {
    PartialJsonParser p;
    std::string json = R"({"key": [val})"; // malformed: invalid token 'val' cannot be completed
    p.feed(std::string_view(json));
    expect_status(p, ParseStatus::Malformed);
    assert(!p.errors().empty());
}

void test_partial_input() {
    // Incomplete JSON object – missing closing brace should be reported as Partial.
    PartialJsonParser p;
    std::string json = R"({"key": "value")";
    p.feed(std::string_view(json));
    expect_status(p, ParseStatus::Partial);
    // Verify the partially parsed object matches expected.
    std::string dump = json_dump(p.root());
    assert(dump == R"({"key":"value"})");
    // No errors should be recorded for a merely incomplete (partial) input.
    assert(p.errors().empty());
}
void test_conversion_helpers() {
    // to_vector
    Array arr;
    arr.emplace_back(Primitive(1.0));
    arr.emplace_back(Primitive(2.0));
    arr.emplace_back(Primitive(3.0));
    Node arrNode(arr);
    auto vec = to_vector(arrNode);
    assert(vec.size() == 3);
    assert(to_primitive<double>(vec[0].as_primitive()) == 1.0);
    assert(to_primitive<double>(vec[2].as_primitive()) == 3.0);

    // to_ordered_map
    OrderedObject obj;
    obj.insert("x", Node(Primitive(10.0)));
    obj.insert("y", Node(Primitive(20.0)));
    Node objNode(obj);
    auto map = to_ordered_map(objNode);
    assert(map.size() == 2);
    assert(to_primitive<double>(map.at("x").as_primitive()) == 10.0);
    assert(to_primitive<double>(map.at("y").as_primitive()) == 20.0);

    // to_primitive
    Node prim = Node(Primitive(42.0));
    int val = to_primitive<int>(prim);
    assert(val == 42);
}

int main() {
    test_complete_object();
    test_key_order_preservation();
    test_incremental_parsing();
    test_partial_input();
    test_malformed_input();
    test_conversion_helpers();
    std::cerr << "All tests passed.\n";
    return 0;
}