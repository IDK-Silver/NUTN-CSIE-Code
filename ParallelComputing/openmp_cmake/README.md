# OpenMP CMake

This folder contains two C/OpenMP examples:

- `parallel_for_cpu_demo`: parallel loop scheduling demo with per-thread CPU/core reporting
- `linear_search_compare`: serial vs. parallel linear search timing comparison

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/parallel_for_cpu_demo 4 8
./build/linear_search_compare 4 1000000 999999 10
```

`parallel_for_cpu_demo` arguments:

```bash
./build/parallel_for_cpu_demo <thread_count> <iteration_count>
./build/parallel_for_cpu_demo --help
```

- `thread_count`: OpenMP thread count
- `iteration_count`: number of loop iterations in the `parallel for`

`linear_search_compare` arguments:

```bash
./build/linear_search_compare 4 1000000 999999 10
./build/linear_search_compare <thread_count> <element_count> <target_value> [repeat_count]
./build/linear_search_compare --help
```

- `thread_count`: OpenMP thread count for the parallel search
- `element_count`: size of the dynamically allocated integer array
- `target_value`: value to search for
- `repeat_count`: optional repetition count used to average timings
