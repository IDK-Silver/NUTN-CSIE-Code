#include "stencil_common.hpp"

#include <iomanip>

__global__ void stencil_1d_shared(const int* in, int* out, int n) {
    extern __shared__ int temp[];

    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    int block_start = blockIdx.x * blockDim.x;
    int local_index = threadIdx.x + STENCIL_RADIUS;
    int padded_size = n + 2 * STENCIL_RADIUS;
    int tile_size = blockDim.x + 2 * STENCIL_RADIUS;

    for (int tile_index = threadIdx.x; tile_index < tile_size; tile_index += blockDim.x) {
        int input_index = block_start + tile_index;
        temp[tile_index] = (input_index < padded_size) ? in[input_index] : 0;
    }

    __syncthreads();

    if (global_index < n) {
        int result = 0;
        for (int offset = -STENCIL_RADIUS; offset <= STENCIL_RADIUS; ++offset) {
            result += temp[local_index + offset];
        }
        out[global_index] = result;
    }
}

void print_usage(const char* program) {
    std::cout << "Usage:\n"
              << "  " << program << " [N] [benchmark_runs] [seed]\n\n";
    print_common_stencil_options();
    std::cout << "This target runs the GPU stencil with dynamic shared memory.\n"
              << "It benchmarks block sizes: 16, 32, 64, 128, 256, 512, 1024.\n";
}

float benchmark_kernel(
    const int* d_in,
    int* d_out,
    int n,
    int block_size,
    int benchmark_runs) {
    int grid_size = (n + block_size - 1) / block_size;
    size_t shared_bytes = static_cast<size_t>(block_size + 2 * STENCIL_RADIUS) * sizeof(int);

    for (int i = 0; i < STENCIL_WARMUP_RUNS; ++i) {
        stencil_1d_shared<<<grid_size, block_size, shared_bytes>>>(d_in, d_out, n);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        stencil_1d_shared<<<grid_size, block_size, shared_bytes>>>(d_in, d_out, n);
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
    if (wants_help(argc, argv)) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    StencilArgs args;
    if (!parse_stencil_args(argc, argv, args)) {
        std::cerr << "Usage: " << argv[0] << " [N] [benchmark_runs] [seed]" << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    std::vector<int> input;
    std::vector<int> expected;
    std::vector<int> output(args.n);
    fill_input(input, args.n, args.seed);
    stencil_cpu_reference(input, expected, args.n);

    const size_t input_bytes = input.size() * sizeof(int);
    const size_t output_bytes = static_cast<size_t>(args.n) * sizeof(int);
    int* d_in = nullptr;
    int* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_in), input_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out), output_bytes));
    CHECK_CUDA(cudaMemcpy(d_in, input.data(), input_bytes, cudaMemcpyHostToDevice));

    std::cout << "1D stencil GPU shared memory\n";
    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "N: " << args.n << '\n';
    std::cout << "Radius: " << STENCIL_RADIUS << '\n';
    std::cout << "Benchmark runs: " << args.benchmark_runs << '\n';
    std::cout << "Timing: kernel only, excluding cudaMemcpy\n\n";

    std::cout << std::left << std::setw(12) << "Block" << std::setw(12) << "Grid"
              << std::setw(18) << "Shared bytes" << std::setw(18) << "Avg time"
              << std::setw(18) << "Checksum" << "Result\n";
    std::cout << std::string(90, '-') << '\n';

    const int block_sizes[] = {16, 32, 64, 128, 256, 512, 1024};
    bool all_ok = true;
    for (int block_size : block_sizes) {
        if (block_size > prop.maxThreadsPerBlock) {
            continue;
        }

        size_t shared_bytes = static_cast<size_t>(block_size + 2 * STENCIL_RADIUS) * sizeof(int);
        CHECK_CUDA(cudaMemset(d_out, 0, output_bytes));
        float avg_ms = benchmark_kernel(d_in, d_out, args.n, block_size, args.benchmark_runs);
        CHECK_CUDA(cudaMemcpy(output.data(), d_out, output_bytes, cudaMemcpyDeviceToHost));
        bool ok = verify_result(expected, output);
        all_ok = all_ok && ok;
        int grid_size = (args.n + block_size - 1) / block_size;

        std::cout << std::left << std::setw(12) << block_size << std::setw(12) << grid_size
                  << std::setw(18) << shared_bytes << std::setw(18)
                  << (std::to_string(avg_ms) + " ms") << std::setw(18) << checksum(output)
                  << (ok ? "OK" : "FAILED") << '\n';
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
