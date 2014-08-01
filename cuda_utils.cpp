#include <cuda_utils.h>

extern "C" void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf((stderr, "Runtime API error %d: %s.\n", (int) err, cudaGetErrorString(err)));
        exit(-1);
    }
    return;
}