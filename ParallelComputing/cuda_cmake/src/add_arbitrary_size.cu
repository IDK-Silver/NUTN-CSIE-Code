#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifndef N
#define N (2048 * 2048 + 1)
#endif

#define DEFAULT_BENCHMARK_RUNS 100
#define DEFAULT_SEED 1234
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
              << "  benchmark_runs  Number of timed runs per version, default "
              << DEFAULT_BENCHMARK_RUNS << '\n'
              << "  seed            Random seed, default " << DEFAULT_SEED << "\n\n"
              << "This target demonstrates arbitrary vector sizes and compares timing:\n"
              << "  index = threadIdx.x + blockIdx.x * blockDim.x\n"
              << "  if (index < n) c[index] = a[index] + b[index]\n"
              << "  launch = <<<(N + M - 1) / M, M>>>\n"
              << "  tested M values: 1, 32, 64, 128, 256, 512, 1024\n";
}

__global__ void add(const int* a, const int* b, int* c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void add_sequential_cpu(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
    for (size_t i = 0; i < a.size(); ++i) {
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

double benchmark_sequential_cpu(
    const std::vector<int>& a,
    const std::vector<int>& b,
    std::vector<int>& c,
    int benchmark_runs) {
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        add_sequential_cpu(a, b, c);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmark_runs; ++i) {
        add_sequential_cpu(a, b, c);
    }
    auto stop = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> total_ms = stop - start;
    return total_ms.count() / benchmark_runs;
}

float benchmark_add_kernel(
    const int* d_a,
    const int* d_b,
    int* d_c,
    int n,
    int threads_per_block,
    int benchmark_runs) {
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;

    for (int i = 0; i < WARMUP_RUNS; ++i) {
        add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, n);
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
    unsigned int seed = DEFAULT_SEED;

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

    const size_t bytes = static_cast<size_t>(n) * sizeof(int);
    std::vector<int> a(n), b(n), c(n);
    random_ints(a, seed);
    random_ints(b, seed + 1);

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Vector size N: " << n << '\n';
    std::cout << "Benchmark runs: " << benchmark_runs << '\n';
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << '\n';
    std::cout << "Timing:\n";
    std::cout << "  CPU sequential: addition loop only\n";
    std::cout << "  CUDA versions: kernel only, excluding cudaMemcpy\n\n";

    std::cout << std::left << std::setw(24) << "Version" << std::setw(22)
              << "<<<grid, block>>>" << std::setw(12) << "M" << std::setw(12)
              << "Blocks" << std::setw(12) << "Covered" << std::setw(12) << "Extra"
              << std::setw(18) << "Avg time" << "Result\n";
    std::cout << std::string(112, '-') << '\n';

    double cpu_ms = benchmark_sequential_cpu(a, b, c, benchmark_runs);
    bool cpu_ok = verify_result(a, b, c);
    std::cout << std::left << std::setw(24) << "CPU sequential" << std::setw(22) << "N/A"
              << std::setw(12) << "N/A" << std::setw(12) << "N/A" << std::setw(12)
              << "N/A" << std::setw(12) << "N/A" << std::setw(18)
              << (std::to_string(cpu_ms) + " ms") << (cpu_ok ? "OK" : "FAILED") << '\n';

    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_a), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_b), bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_c), bytes));
    CHECK_CUDA(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    const int test_threads_per_block[] = {1, 32, 64, 128, 256, 512, 1024};
    bool all_ok = cpu_ok;

    for (int threads_per_block : test_threads_per_block) {
        if (threads_per_block > prop.maxThreadsPerBlock) {
            continue;
        }

        int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
        if (blocks_per_grid > prop.maxGridSize[0]) {
            std::cout << std::left << std::setw(24) << "CUDA guarded" << std::setw(22)
                      << "SKIP" << std::setw(12) << threads_per_block << std::setw(12)
                      << blocks_per_grid << std::setw(12) << "N/A" << std::setw(12)
                      << "N/A" << std::setw(18) << "N/A" << "grid too large\n";
            continue;
        }

        int covered_threads = blocks_per_grid * threads_per_block;
        int guarded_threads = covered_threads - n;
        float avg_ms =
            benchmark_add_kernel(d_a, d_b, d_c, n, threads_per_block, benchmark_runs);
        CHECK_CUDA(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
        bool ok = verify_result(a, b, c);
        all_ok = all_ok && ok;

        std::string launch_config = "<<<" + std::to_string(blocks_per_grid) + ", " +
                                    std::to_string(threads_per_block) + ">>>";
        std::cout << std::left << std::setw(24) << "CUDA guarded" << std::setw(22)
                  << launch_config << std::setw(12) << threads_per_block << std::setw(12)
                  << blocks_per_grid << std::setw(12) << covered_threads << std::setw(12)
                  << guarded_threads << std::setw(18) << (std::to_string(avg_ms) + " ms")
                  << (ok ? "OK" : "FAILED") << '\n';
    }

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
