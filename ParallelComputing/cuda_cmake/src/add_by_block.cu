#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef N
#define N 1024
#endif

#define DEFAULT_BENCHMARK_RUNS 1000
#define WARMUP_RUNS 10

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

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " [N] [benchmark_runs] [seed]\n\n"
              << "Options:\n"
              << "  N               Vector size, default " << N << '\n'
              << "  benchmark_runs  Number of timed kernel launches, default "
              << DEFAULT_BENCHMARK_RUNS << '\n'
              << "  seed            Random seed, default 1234\n\n"
              << "This target uses blockIdx.x with launch config <<<N, 1>>>.\n";
}

__global__ void add_by_block(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

void random_ints(std::vector<int>& values, unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 1000);
    for (int& value : values) {
        value = dist(rng);
    }
}

bool verify_result(const std::vector<int>& a, const std::vector<int>& b, const std::vector<int>& c) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " + " << b[i]
                      << " != " << c[i] << std::endl;
            return false;
        }
    }
    return true;
}

float benchmark_kernel(const int* d_a, const int* d_b, int* d_c, int n, int benchmark_runs) {
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        add_by_block<<<n, 1>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        add_by_block<<<n, 1>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / benchmark_runs;
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    int n = N;
    int benchmark_runs = DEFAULT_BENCHMARK_RUNS;
    unsigned int seed = 1234;

    if (argc >= 2) {
        n = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        benchmark_runs = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        seed = static_cast<unsigned int>(std::strtoul(argv[3], nullptr, 10));
    }
    if (n <= 0 || benchmark_runs <= 0) {
        std::cerr << "Usage: " << argv[0] << " [N] [benchmark_runs] [seed]" << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (n > prop.maxGridSize[0]) {
        std::cerr << "add_by_block launches <<<N, 1>>> and requires N <= maxGridSize[0].\n"
                  << "Current N = " << n << ", maxGridSize[0] = " << prop.maxGridSize[0]
                  << std::endl;
        return EXIT_FAILURE;
    }

    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    std::vector<int> a(n), b(n), c(n);
    random_ints(a, seed);
    random_ints(b, seed + 1);

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));
    CHECK_CUDA(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    float avg_ms = benchmark_kernel(d_a, d_b, d_c, n, benchmark_runs);
    CHECK_CUDA(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    bool ok = verify_result(a, b, c);

    std::string launch_config = "<<<" + std::to_string(n) + ", 1>>>";
    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Vector size N: " << n << '\n';
    std::cout << "Benchmark runs: " << benchmark_runs << '\n';
    std::cout << "Timing: kernel only, excluding cudaMemcpy\n\n";
    std::cout << std::left << std::setw(24) << "Version" << std::setw(20) << "Launch config"
              << std::setw(18) << "Index variable" << std::setw(18) << "Avg kernel time"
              << "Result\n";
    std::cout << std::string(88, '-') << '\n';
    std::cout << std::left << std::setw(24) << "Block-based add" << std::setw(20)
              << launch_config << std::setw(18) << "blockIdx.x" << std::setw(18)
              << (std::to_string(avg_ms) + " ms") << (ok ? "OK" : "FAILED") << '\n';

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
