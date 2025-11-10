# JSON Parser (`json_parser`)

## Overview

The **json_parser** component provides a header‑only implementation for:

- **Ordered objects** – deterministic iteration order for JSON objects.
- **Monotonic partial parsing** – incremental feeding of JSON data without backtracking.
- **Conversion utilities** – easy conversion between JSON and native C++ types.

These utilities are self‑contained and depend only on the C++ standard library.

## API Summary

| Header | Primary symbols |
|--------|-----------------|
| `common/json/ordered_json.hpp` | `OrderedObject`, `ordered_dump()` |
| `common/json/conversion.hpp`   | `json_to_*`, `*_to_json` helpers |
| `common/json/partial_parser.hpp` | `PartialJsonParser`, `ParseStatus`, `json_dump()` |

All headers are located under `common/json/` and are included via the `json_parser` target.

## Usage Example

```cpp
#include <ordered_json.hpp>
#include <partial_parser.hpp>

int main() {
    PartialJsonParser parser;
    std::string json = R"({ "a": 1, "b": [true, false] })";

    // Feed the JSON incrementally
    for (char c : json) {
        parser.feed(std::string_view(&c, 1));
    }

    if (parser.status() == ParseStatus::Complete) {
        std::cout << "Parsed object: " << json_dump(parser.root()) << std::endl;
    }
}
```

## Build

The `json_parser` target is an INTERFACE library. It is automatically added to the project when you build.

```bash
# Build the project
cmake -S . -B build && cmake --build build

# Run the JSON parser test
ctest -R test-json-parser
```

## Extensibility

The parser is built on a generic `Grammar` abstraction, allowing future extensions to support non‑JSON grammars with minimal changes.
