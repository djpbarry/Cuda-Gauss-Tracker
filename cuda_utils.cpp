#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" void checkCudaError();

extern "C" void checkCudaError(){
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf((stderr, "Runtime API error %d: %s.\n", (int)err, cudaGetErrorString( err ) ));
        exit(-1);
	}
	return;
}