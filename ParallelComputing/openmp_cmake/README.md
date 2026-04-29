# OpenMP CMake

This folder contains four C/OpenMP examples:

- `parallel_for_cpu_demo`: parallel loop scheduling demo with per-thread CPU/core reporting
- `linear_search_compare`: serial vs. parallel linear search timing comparison
- `prime_schedule_benchmark`: static, dynamic, and guided schedule timing comparison
- `matrix_multiply_compare`: sequential vs. OpenMP matrix multiplication timing comparison

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/parallel_for_cpu_demo 4 8
./build/linear_search_compare 4 1000000 999999 10
./build/prime_schedule_benchmark 4 1000000 8 5
./build/matrix_multiply_compare 4 256 3
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

`prime_schedule_benchmark` arguments:

```bash
./build/prime_schedule_benchmark 4 1000000 8 5
./build/prime_schedule_benchmark <thread_count> <max_number> <chunk_size> [repeat_count]
./build/prime_schedule_benchmark --help
```

- `thread_count`: OpenMP thread count for the benchmark
- `max_number`: count prime numbers from 2 to this value
- `chunk_size`: chunk size used by `static`, `dynamic`, and `guided`
- `repeat_count`: optional repetition count used to average timings

`matrix_multiply_compare` arguments:

```bash
./build/matrix_multiply_compare 4 256 3
./build/matrix_multiply_compare <thread_count> <matrix_size> [repeat_count]
./build/matrix_multiply_compare --help
```

- `thread_count`: OpenMP thread count for the parallel multiplication cases
- `matrix_size`: matrix dimension N for NxN matrix multiplication
- `repeat_count`: optional repetition count used to average timings
