#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2 || (argc > 1 && std::strcmp(argv[1], "--help") == 0)) {
        std::cout << "Usage:\n"
                  << "  ./build/vector_add <N> [repeat_count]\n"
                  << "  <N>: vector size\n"
                  << "  [repeat_count]: repeat times, default 10\n";
        return 0;
    }

    int n = std::atoi(argv[1]);
    if (n <= 0) {
        std::cerr << "Error: N must be a positive integer.\n";
        return 1;
    }
    int repeat = (argc >= 3) ? std::atoi(argv[2]) : 10;
    repeat = std::max(1, repeat);

    std::vector<float> h_a(n), h_b(n), h_c_cpu(n), h_c_gpu(n);
    for (int i = 0; i < n; i++) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = std::sqrt(static_cast<float>(i));
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < repeat; r++) {
        for (int i = 0; i < n; i++) {
            h_c_cpu[i] = h_a[i] + h_b[i];
        }
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count() / repeat;

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    cudaError_t err = cudaMalloc(&d_a, sizeof(float) * n);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_a: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    err = cudaMalloc(&d_b, sizeof(float) * n);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_b: " << cudaGetErrorString(err) << '\n';
        cudaFree(d_a);
        return 1;
    }
    err = cudaMalloc(&d_c, sizeof(float) * n);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed for d_c: " << cudaGetErrorString(err) << '\n';
        cudaFree(d_a);
        cudaFree(d_b);
        return 1;
    }

    err = cudaMemcpy(d_a, h_a.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D d_a failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }
    err = cudaMemcpy(d_b, h_b.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D d_b failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsed = 0.0f;

    constexpr int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    for (int r = 0; r < repeat; r++) {
        cudaEventRecord(start);
        vector_add<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, end);
        elapsed += ms;
    }

    err = cudaMemcpy(h_c_gpu.data(), d_c, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H d_c failed: " << cudaGetErrorString(err) << '\n';
        return 1;
    }

    for (int i = 0; i < n; i++) {
        if (std::fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-4f) {
            std::cerr << "Mismatch at index " << i << ": CPU=" << h_c_cpu[i] << " GPU="
                      << h_c_gpu[i] << '\n';
            break;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "N=" << n << '\n';
    std::cout << "CPU avg ms: " << cpu_ms << '\n';
    std::cout << "CUDA kernel avg ms: " << (elapsed / repeat) << '\n';
    return 0;
}
