#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define STENCIL_RADIUS 256
#define DEFAULT_STENCIL_N (1 << 24)
#define DEFAULT_STENCIL_RUNS 3
#define DEFAULT_STENCIL_SEED 1234
#define STENCIL_WARMUP_RUNS 1

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << "\nFile: " << __FILE__ << "\nLine: " << __LINE__      \
                      << std::endl;                                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

struct StencilArgs {
    int n = DEFAULT_STENCIL_N;
    int benchmark_runs = DEFAULT_STENCIL_RUNS;
    unsigned int seed = DEFAULT_STENCIL_SEED;
};

inline bool wants_help(int argc, char** argv) {
    return argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h");
}

inline bool parse_stencil_args(int argc, char** argv, StencilArgs& args) {
    if (argc >= 2) {
        args.n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        args.benchmark_runs = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        args.seed = static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10));
    }
    return args.n > 0 && args.benchmark_runs > 0;
}

inline void print_common_stencil_options() {
    std::cout << "Options:\n"
              << "  N               Number of output elements, default " << DEFAULT_STENCIL_N
              << '\n'
              << "  benchmark_runs  Number of timed runs, default " << DEFAULT_STENCIL_RUNS
              << '\n'
              << "  seed            Random seed, default " << DEFAULT_STENCIL_SEED << "\n\n"
              << "The input array has N + 2 * RADIUS elements and zero padding on both sides.\n"
              << "RADIUS = " << STENCIL_RADIUS << '\n';
}

inline void fill_input(std::vector<int>& input, int n, unsigned int seed) {
    input.assign(n + 2 * STENCIL_RADIUS, 0);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1000);
    for (int i = STENCIL_RADIUS; i < n + STENCIL_RADIUS; ++i) {
        input[i] = dist(rng);
    }
}

inline void stencil_cpu_reference(const std::vector<int>& input, std::vector<int>& output, int n) {
    output.resize(n);
    for (int i = 0; i < n; ++i) {
        int sum = 0;
        int center = i + STENCIL_RADIUS;
        for (int offset = -STENCIL_RADIUS; offset <= STENCIL_RADIUS; ++offset) {
            sum += input[center + offset];
        }
        output[i] = sum;
    }
}

inline bool verify_result(const std::vector<int>& expected, const std::vector<int>& actual) {
    if (expected.size() != actual.size()) {
        std::cerr << "Size mismatch: expected " << expected.size() << ", got " << actual.size()
                  << std::endl;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected[i]
                      << ", got " << actual[i] << std::endl;
            return false;
        }
    }
    return true;
}

inline unsigned long long checksum(const std::vector<int>& values) {
    unsigned long long sum = 0;
    for (int value : values) {
        sum += static_cast<unsigned int>(value);
    }
    return sum;
}

inline double benchmark_stencil_cpu(
    const std::vector<int>& input,
    std::vector<int>& output,
    int n,
    int benchmark_runs) {
    for (int i = 0; i < STENCIL_WARMUP_RUNS; ++i) {
        stencil_cpu_reference(input, output, n);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        stencil_cpu_reference(input, output, n);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_ms = stop - start;
    return total_ms.count() / benchmark_runs;
}
