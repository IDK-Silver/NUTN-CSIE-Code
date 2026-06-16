#include <cuda_runtime.h>

#include <cstdio>

__global__ void magic(char* dptr) {
    for (int i = 0; i < 13; i++) {
        dptr[i]++;
    }
}

int main() {
    char gu[13] = {'G', 'd', 'k', 'k', 'n', 31, 'V', 'n', 'q', 'k', 'c', 32, -1};
    char* dptr = nullptr;

    cudaError_t err = cudaMalloc(&dptr, sizeof(char) * 13);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(dptr, gu, sizeof(char) * 13, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(dptr);
        return 1;
    }

    magic<<<1, 1>>>(dptr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "magic kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(dptr);
        return 1;
    }

    err = cudaMemcpy(gu, dptr, sizeof(char) * 13, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(dptr);
        return 1;
    }

    std::printf("%s\n", gu);

    cudaFree(dptr);
    return 0;
}
