#include <cuda_runtime.h>

#include <cstdio>

void print_dev_prop(const cudaDeviceProp& dev_prop) {
    std::printf("Major revision number: %d\n", dev_prop.major);
    std::printf("Minor revision number: %d\n", dev_prop.minor);
    std::printf("Name: %s\n", dev_prop.name);
    std::printf("Total global memory: %zu\n", dev_prop.totalGlobalMem);
    std::printf("Total shared memory per block: %zu\n", dev_prop.sharedMemPerBlock);
    std::printf("Total registers per block: %d\n", dev_prop.regsPerBlock);
    std::printf("Warp size: %d\n", dev_prop.warpSize);
    std::printf("Maximum memory pitch: %zu\n", dev_prop.memPitch);
    std::printf("Maximum threads per block: %d\n", dev_prop.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) {
        std::printf("Maximum dimension %d of block: %d\n", i, dev_prop.maxThreadsDim[i]);
    }
    for (int i = 0; i < 3; ++i) {
        std::printf("Maximum dimension %d of grid: %d\n", i, dev_prop.maxGridSize[i]);
    }
    std::printf("Clock rate: %d\n", dev_prop.clockRate);
    std::printf("Total constant memory: %zu\n", dev_prop.totalConstMem);
    std::printf("Texture alignment: %zu\n", dev_prop.textureAlignment);
    std::printf("Concurrent copy and execution: %s\n", dev_prop.deviceOverlap ? "Yes" : "No");
    std::printf("Number of multiprocessors: %d\n", dev_prop.multiProcessorCount);
    std::printf(
        "Kernel execution timeout: %s\n",
        dev_prop.kernelExecTimeoutEnabled ? "Yes" : "No");
}

int main() {
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    std::printf("CUDA Device Query...\n");
    std::printf("There are %d CUDA devices.\n", dev_count);

    for (int i = 0; i < dev_count; ++i) {
        std::printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp dev_prop{};
        err = cudaGetDeviceProperties(&dev_prop, i);
        if (err != cudaSuccess) {
            std::fprintf(
                stderr,
                "cudaGetDeviceProperties failed for device %d: %s\n",
                i,
                cudaGetErrorString(err));
            return 1;
        }
        print_dev_prop(dev_prop);
    }

    return 0;
}
