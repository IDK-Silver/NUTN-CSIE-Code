#include <cuda_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define DEFAULT_ARRAY_SIZE (1 << 22)
#define DEFAULT_BLOCK_SIZE 256
#define DEFAULT_BENCHMARK_RUNS 100
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
              << "  " << program << " [array_size] [block_size] [benchmark_runs]\n\n"
              << "Options:\n"
              << "  array_size      Number of float elements, default " << DEFAULT_ARRAY_SIZE
              << '\n'
              << "  block_size      Threads per block, default " << DEFAULT_BLOCK_SIZE << '\n'
              << "  benchmark_runs  Number of timed add/mul pairs, default "
              << DEFAULT_BENCHMARK_RUNS << "\n\n"
              << "This target compares two launches:\n"
              << "  without explicit streams: addKernel then mulKernel in the default stream\n"
              << "  with streams: addKernel in stream1 and mulKernel in stream2\n";
}

__global__ void addKernel(float* d_out, const float* d_in1, const float* d_in2, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_out[idx] = d_in1[idx] + d_in2[idx];
    }
}

__global__ void mulKernel(float* d_out, const float* d_in1, const float* d_in2, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        d_out[idx] = d_in1[idx] * d_in2[idx];
    }
}

bool verify_results(
    const std::vector<float>& in1,
    const std::vector<float>& in2,
    const std::vector<float>& out_add,
    const std::vector<float>& out_mul) {
    for (size_t i = 0; i < in1.size(); ++i) {
        float expected_add = in1[i] + in2[i];
        float expected_mul = in1[i] * in2[i];
        if (out_add[i] != expected_add || out_mul[i] != expected_mul) {
            std::cerr << "Mismatch at index " << i << ": add=" << out_add[i]
                      << " expected " << expected_add << ", mul=" << out_mul[i]
                      << " expected " << expected_mul << std::endl;
            return false;
        }
    }
    return true;
}

void print_values(const char* title, const std::vector<float>& values) {
    std::cout << title << '\n';
    int count = static_cast<int>(values.size());
    int shown = count < 16 ? count : 16;
    for (int i = 0; i < shown; ++i) {
        std::cout << std::setw(8) << values[i];
    }
    if (count > shown) {
        std::cout << " ...";
    }
    std::cout << '\n';
}

void clear_outputs(float* d_out_add, float* d_out_mul, size_t bytes) {
    CHECK_CUDA(cudaMemset(d_out_add, 0, bytes));
    CHECK_CUDA(cudaMemset(d_out_mul, 0, bytes));
    CHECK_CUDA(cudaDeviceSynchronize());
}

float benchmark_default_stream(
    float* d_out_add,
    float* d_out_mul,
    const float* d_in1,
    const float* d_in2,
    int array_size,
    int grid_size,
    int block_size,
    int benchmark_runs) {
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        addKernel<<<grid_size, block_size>>>(d_out_add, d_in1, d_in2, array_size);
        mulKernel<<<grid_size, block_size>>>(d_out_mul, d_in1, d_in2, array_size);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < benchmark_runs; ++i) {
        addKernel<<<grid_size, block_size>>>(d_out_add, d_in1, d_in2, array_size);
        mulKernel<<<grid_size, block_size>>>(d_out_mul, d_in1, d_in2, array_size);
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

float benchmark_two_streams(
    float* d_out_add,
    float* d_out_mul,
    const float* d_in1,
    const float* d_in2,
    int array_size,
    int grid_size,
    int block_size,
    int benchmark_runs,
    cudaStream_t stream1,
    cudaStream_t stream2,
    cudaStream_t timer_stream) {
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        addKernel<<<grid_size, block_size, 0, stream1>>>(d_out_add, d_in1, d_in2, array_size);
        mulKernel<<<grid_size, block_size, 0, stream2>>>(d_out_mul, d_in1, d_in2, array_size);
    }
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream2));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEvent_t done1 = nullptr;
    cudaEvent_t done2 = nullptr;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventCreate(&done1));
    CHECK_CUDA(cudaEventCreate(&done2));

    CHECK_CUDA(cudaEventRecord(start, timer_stream));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, start, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, start, 0));

    for (int i = 0; i < benchmark_runs; ++i) {
        addKernel<<<grid_size, block_size, 0, stream1>>>(d_out_add, d_in1, d_in2, array_size);
        mulKernel<<<grid_size, block_size, 0, stream2>>>(d_out_mul, d_in1, d_in2, array_size);
    }
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(done1, stream1));
    CHECK_CUDA(cudaEventRecord(done2, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(timer_stream, done1, 0));
    CHECK_CUDA(cudaStreamWaitEvent(timer_stream, done2, 0));
    CHECK_CUDA(cudaEventRecord(stop, timer_stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(done1));
    CHECK_CUDA(cudaEventDestroy(done2));
    return total_ms / benchmark_runs;
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    int array_size = DEFAULT_ARRAY_SIZE;
    int block_size = DEFAULT_BLOCK_SIZE;
    int benchmark_runs = DEFAULT_BENCHMARK_RUNS;
    if (argc >= 2) {
        array_size = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        block_size = std::atoi(argv[2]);
    }
    if (argc >= 4) {
        benchmark_runs = std::atoi(argv[3]);
    }
    if (array_size <= 0 || block_size <= 0 || benchmark_runs <= 0) {
        std::cerr << "Usage: " << argv[0] << " [array_size] [block_size] [benchmark_runs]"
                  << std::endl;
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    if (block_size > prop.maxThreadsPerBlock) {
        std::cerr << "block_size must be <= maxThreadsPerBlock.\n"
                  << "Current block_size = " << block_size
                  << ", maxThreadsPerBlock = " << prop.maxThreadsPerBlock << std::endl;
        return EXIT_FAILURE;
    }

    int grid_size = (array_size + block_size - 1) / block_size;
    const size_t array_bytes = static_cast<size_t>(array_size) * sizeof(float);

    std::vector<float> h_in1(array_size);
    std::vector<float> h_in2(array_size);
    std::vector<float> h_out_add(array_size);
    std::vector<float> h_out_mul(array_size);

    for (int i = 0; i < array_size; ++i) {
        h_in1[i] = static_cast<float>(i);
        h_in2[i] = static_cast<float>(i * 2);
    }

    float* d_in1 = nullptr;
    float* d_in2 = nullptr;
    float* d_out_add = nullptr;
    float* d_out_mul = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_in1), array_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_in2), array_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out_add), array_bytes));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out_mul), array_bytes));

    CHECK_CUDA(cudaMemcpy(d_in1, h_in1.data(), array_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_in2, h_in2.data(), array_bytes, cudaMemcpyHostToDevice));

    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;
    cudaStream_t timer_stream = nullptr;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&timer_stream, cudaStreamNonBlocking));

    clear_outputs(d_out_add, d_out_mul, array_bytes);
    float default_ms = benchmark_default_stream(
        d_out_add, d_out_mul, d_in1, d_in2, array_size, grid_size, block_size, benchmark_runs);
    CHECK_CUDA(cudaMemcpy(h_out_add.data(), d_out_add, array_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_mul.data(), d_out_mul, array_bytes, cudaMemcpyDeviceToHost));
    bool default_ok = verify_results(h_in1, h_in2, h_out_add, h_out_mul);

    clear_outputs(d_out_add, d_out_mul, array_bytes);
    float stream_ms = benchmark_two_streams(
        d_out_add,
        d_out_mul,
        d_in1,
        d_in2,
        array_size,
        grid_size,
        block_size,
        benchmark_runs,
        stream1,
        stream2,
        timer_stream);
    CHECK_CUDA(cudaMemcpy(h_out_add.data(), d_out_add, array_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_mul.data(), d_out_mul, array_bytes, cudaMemcpyDeviceToHost));
    bool stream_ok = verify_results(h_in1, h_in2, h_out_add, h_out_mul);

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Concurrent kernels supported: " << (prop.concurrentKernels ? "yes" : "no")
              << '\n';
    std::cout << "Array size: " << array_size << '\n';
    std::cout << "Launch config: <<<" << grid_size << ", " << block_size << ">>>\n";
    std::cout << "Benchmark runs: " << benchmark_runs << '\n';
    std::cout << "Timing: two kernels per run, excluding cudaMemcpy\n\n";

    std::cout << std::left << std::setw(28) << "Mode" << std::setw(30) << "Kernel placement"
              << std::setw(18) << "Avg pair time" << "Result\n";
    std::cout << std::string(88, '-') << '\n';
    std::cout << std::left << std::setw(28) << "Without explicit streams" << std::setw(30)
              << "default stream: add then mul" << std::setw(18)
              << (std::to_string(default_ms) + " ms") << (default_ok ? "OK" : "FAILED")
              << '\n';
    std::cout << std::left << std::setw(28) << "With two streams" << std::setw(30)
              << "stream1 add, stream2 mul" << std::setw(18)
              << (std::to_string(stream_ms) + " ms") << (stream_ok ? "OK" : "FAILED")
              << '\n';
    if (stream_ms > 0.0f) {
        std::cout << "\nSpeedup without/with streams: " << (default_ms / stream_ms) << "x\n";
    }

    std::cout << '\n';
    print_values("Addition results:", h_out_add);
    print_values("Multiplication results:", h_out_mul);

    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(timer_stream));
    CHECK_CUDA(cudaFree(d_in1));
    CHECK_CUDA(cudaFree(d_in2));
    CHECK_CUDA(cudaFree(d_out_add));
    CHECK_CUDA(cudaFree(d_out_mul));
    return (default_ok && stream_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
