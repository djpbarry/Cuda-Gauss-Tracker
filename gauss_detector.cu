#include <matrix.h>
//#include <cutil_inline.h>
#include <math.h>
#include <stdio.h>
//#include <device_functions.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_REPS 1

extern "C" float GaussFitter(Matrix A, int maxcount, float sigEst, float maxThresh);

void checkCudaError();

__shared__ float _2sig2, _maxThresh, xyStepSize, magStepSize;

__global__ void  GaussFitterKernel(Matrix A, float sigEst, int reps); 

__device__ float getRSquared(int x0, float srs, charMatrix M);

__device__ void centreOfMass(float *x, float *y, int index, charMatrix image);

__device__ void doMultiFit(charMatrix M, int x0, int N, float *xe, float *ye, float *mag, float *r);

__device__ float sumMultiResiduals(int x0, float *xe, float *ye, float *mag, charMatrix M, float xinc, float yinc, float minc, int index, int N);

__device__ float multiEvaluate(float x0, float y0, float mag, int x, int y);

__device__ int initialiseFitting(charMatrix image, int index, float *xe, float *ye, float *mag);

 __device__ float getRSquared(int x0, float srs, charMatrix M) {
	int y0 = FIT_RADIUS;
    float sumZ = 0.0f;
#pragma unroll FIT_SIZE
    for (int y = y0 - FIT_RADIUS; y <= y0 + FIT_RADIUS; y++) {
		int offset = y * M.stride;
#pragma unroll FIT_SIZE
        for (int x = x0 - FIT_RADIUS; x <= x0 + FIT_RADIUS; x++) {
            sumZ += M.elements[x + offset];
        }
    }
    float mean = __fdividef(sumZ, FIT_SIZE);
    float sumMeanDiffSqr = 0.0f;
#pragma unroll FIT_SIZE
    for (int y = y0 - FIT_RADIUS; y <= y0 + FIT_RADIUS; y++) {
		int offset = y * M.stride;
#pragma unroll FIT_SIZE
        for (int x = x0 - FIT_RADIUS; x <= x0 + FIT_RADIUS; x++) {
            sumMeanDiffSqr += (M.elements[x + offset] - mean) * (M.elements[x + offset] - mean);
        }
    }
    return 1.0f - __fdividef(srs, sumMeanDiffSqr);
}

extern "C" float GaussFitter(Matrix A, int maxcount, float sigEst, float maxThresh) 
{ 
	cudaSetDevice(0);
	checkCudaError();
	cudaEvent_t start, stop;
	float outerTime, innerTime, copyToDevice, copyFromDevice;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	checkCudaError();
    Matrix d_A; 
    d_A.width = A.width;
	d_A.height = A.height; 
	d_A.stride= A.stride;
	d_A.size = d_A.width * d_A.height;
    size_t matrixsize = A.width * A.height * sizeof(float); 
	cudaEventRecord(start, 0);
	checkCudaError();
	cudaMalloc(&d_A.elements, matrixsize);
	checkCudaError();
	cudaMemcpy(d_A.elements, A.elements, matrixsize, cudaMemcpyHostToDevice);
	checkCudaError();
	cudaEventRecord(stop, 0);
	checkCudaError();
	cudaEventSynchronize(stop);
	checkCudaError();
	cudaEventElapsedTime(&copyToDevice, start, stop);
	checkCudaError();

	printf("\n\nCopy To Device: %.0f\n", copyToDevice);

	int numblocks = maxcount / BLOCK_SIZE_X;
	if (numblocks * BLOCK_SIZE_X < maxcount) numblocks++;
	if (numblocks % 2 != 0) numblocks++;

	printf("\n\nMax Count: %d\nNumber of Blocks: %d\n", maxcount, numblocks);

    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid(numblocks, 1);

	cudaEventRecord(start, 0);
	checkCudaError();
	for(int i=0; i<NUM_REPS; i++){
		GaussFitterKernel<<<dimGrid, dimBlock>>>(d_A, sigEst, 1);
	}
	cudaEventRecord(stop, 0);
	checkCudaError();
	cudaEventSynchronize(stop);
	checkCudaError();
	cudaEventElapsedTime(&outerTime, start, stop);
	checkCudaError();
	
	cudaEventRecord(start, 0);
	checkCudaError();

	GaussFitterKernel<<<dimGrid, dimBlock>>>(d_A, sigEst, NUM_REPS);
	cudaEventRecord(stop, 0);
	checkCudaError();
	cudaEventSynchronize(stop);
	checkCudaError();
	cudaEventElapsedTime(&innerTime, start, stop);
	checkCudaError();

	checkCudaError();
	
	cudaEventRecord(start, 0);
	checkCudaError();
	cudaMemcpy(A.elements, d_A.elements, matrixsize, cudaMemcpyDeviceToHost);
	checkCudaError();

	cudaEventRecord(stop, 0);
	checkCudaError();
	cudaEventSynchronize(stop);
	checkCudaError();
	cudaEventElapsedTime(&copyFromDevice, start, stop);
	checkCudaError();

	cudaFree(d_A.elements);
	checkCudaError();

	cudaDeviceReset();
	checkCudaError();

	printf("\n\nCopy From Device: %.0f\n", copyFromDevice);

	printf("\nInner Time: %.0f Outer Time: %.0f\n", innerTime / NUM_REPS, outerTime / NUM_REPS);

	return (outerTime + innerTime) / (NUM_REPS * 2.0f);
} 
 
__global__ void GaussFitterKernel(Matrix A, float sigEst, int reps)
{
	_2sig2 = 2.0f * sigEst * sigEst;
	xyStepSize = __fdividef(XY_STEP_SIZE, STEP_TOL);
	magStepSize = __fdividef(MAG_STEP_SIZE, STEP_TOL);
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    //int xRegionCentre = (blockIdx.x * blockDim.x + threadIdx.x) * FIT_SIZE + FIT_RADIUS;
	for(int m=0; m<reps; m++){
		__shared__ char AsElements[BLOCK_SIZE_X * BLOCK_SIZE_Y * FIT_SIZE * FIT_SIZE];
		charMatrix As;
		As.width = blockDim.x * FIT_SIZE;
		As.height = blockDim.y * FIT_SIZE;
		As.stride = As.width;
		As.elements = &AsElements[0];
		int start = threadIdx.x*FIT_SIZE;
		int stop = threadIdx.x*FIT_SIZE+FIT_SIZE;
		int blockOffset = blockIdx.x * blockDim.x * FIT_SIZE;
		for(int j=0; j<FIT_SIZE; j++){
			int soffset = j * As.stride;
			int goffset = (j+HEADER) * A.stride + blockOffset;
			for(int i=start; i<stop; i++){
				As.elements[soffset + i] = (char)floor((A.elements[goffset + i]));
			}
		}
		__syncthreads();
		float xe[N_MAX * N_MAX];
		float ye[N_MAX * N_MAX];
		float mag[N_MAX * N_MAX];
		int xRegionCentre = threadIdx.x * FIT_SIZE + FIT_RADIUS;
		int best = initialiseFitting(As, xRegionCentre, xe, ye, mag);
		A.elements[index + A.stride * BEST_ROW] = best;
		for(int j=0; j<=best; j++){
			A.elements[index + A.stride * (XE_ROW + j)] = xe[N_MAX * best + j]+blockOffset;
			A.elements[index + A.stride * (YE_ROW + j)] = ye[N_MAX * best + j]+HEADER;
			A.elements[index + A.stride * (MAG_ROW + j)] = mag[N_MAX * best + j];
		}
		__syncthreads();
	}
}

__device__ int initialiseFitting(charMatrix image, int index, float *xe, float *ye, float *mag){
	centreOfMass(&xe[0], &ye[0], index, image);
	//mag[0] = image.elements[index + (FIT_RADIUS + HEADER) * image.stride];
	mag[0] = image.elements[index + FIT_RADIUS * image.stride];
	float r[N_MAX];
	doMultiFit(image, index, 0, xe, ye, mag, r);
	for(int n=1; n<N_MAX; n++){
		int noffset = n * N_MAX;
		mag[n + noffset] = 0.0f;
		xe[n + noffset] = 0.0f;
		ye[n + noffset] = 0.0f;
		//for(int j=HEADER; j<FIT_SIZE; j++){
		for(int j=0; j<FIT_SIZE; j++){
			int ioffset = j * image.stride;
			for(int i=index - FIT_RADIUS; i<index - FIT_RADIUS+FIT_SIZE; i++){
				float residual = image.elements[i + ioffset];
				for(int m=0; m<n; m++){
					xe[m + noffset] = xe[m + (n-1) * N_MAX];
					ye[m + noffset] = ye[m + (n-1) * N_MAX];
					mag[m + noffset] = mag[m + (n-1) * N_MAX];
					residual -= multiEvaluate(xe[m + noffset], ye[m + noffset], mag[m + noffset], i, j);
				}
				if(residual > mag[n + noffset]){
					mag[n + noffset] = residual;
					xe[n + noffset] = i;
					ye[n + noffset] = j;
				}
			}
		}
		doMultiFit(image, index, n, xe, ye, mag, r);
	}
	int best = -1;
	float max = 0.0f;
	for(int i=0; i<N_MAX; i++){
		if(r[i] > max){
			max = r[i];
			best = i;
		}
	}
	return best;
}

__device__ void centreOfMass(float *x, float *y, int index, charMatrix image){
	float xsum = 0.0f;
	float ysum = 0.0f;
	float sum = 0.0f;
	//for(int j = HEADER; j<HEADER + FIT_SIZE; j++){
	for(int j = 0; j<FIT_SIZE; j++){
		int offset = j * image.stride;
		for(int i=index-FIT_RADIUS; i<=index+FIT_RADIUS; i++){
			xsum += i * image.elements[i + offset];
			ysum += j * image.elements[i + offset];
			sum += image.elements[i + offset];
		}
	}
	*x = __fdividef(xsum, sum);
	*y = __fdividef(ysum, sum);
}

__device__ void doMultiFit(charMatrix M, int x0, int N, float *xe, float *ye, float *mag, float *r){
	int nnmax = N * N_MAX;
	for(int i=0; i<ITERATIONS; i++){
		for(int j=0; j<=N; j++){
			float r1 = sumMultiResiduals(x0, xe, ye, mag, M, -XY_STEP_SIZE, 0.0f, 0.0f, j, N);
			float r2 = sumMultiResiduals(x0, xe, ye, mag, M, XY_STEP_SIZE, 0.0f, 0.0f, j, N);
			float r3 = sumMultiResiduals(x0, xe, ye, mag, M, 0.0f, -XY_STEP_SIZE, 0.0f, j, N);
			float r4 = sumMultiResiduals(x0, xe, ye, mag, M, 0.0f, XY_STEP_SIZE, 0.0f, j, N);
			float r5 = sumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, -MAG_STEP_SIZE, j, N);
			float r6 = sumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, MAG_STEP_SIZE, j, N);
			xe[nnmax + j] -= (r2 - r1) * xyStepSize;
			ye[nnmax + j] -= (r4 - r3) * xyStepSize;
			mag[nnmax + j] -= (r6 - r5) * magStepSize;
			if(mag[nnmax + j] < 0.0f) mag[nnmax + j] = 0.0f;
		}
	}
	r[N] = getRSquared(x0, sumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, 0.0f, 0, N), M);
	return;
}

__device__ float sumMultiResiduals(int x0, float *xe, float *ye, float *mag, charMatrix M, float xinc, float yinc, float minc, int index, int N) {
    float residuals = 0.0f;
	int eoffset = N * N_MAX;
    //for (int j = HEADER; j < HEADER + FIT_SIZE; j++) {
	for (int j = 0; j < FIT_SIZE; j++) {
	int offset = j * M.stride;
    for (int i = x0 - FIT_RADIUS; i <= x0 + FIT_RADIUS; i++) {
		float res = 0.0f;
		int k;
		for(k=0; k < index; k++){
			res += multiEvaluate(xe[k + eoffset], ye[k + eoffset], mag[k + eoffset], i, j);
		}
		res += multiEvaluate(xe[k + eoffset] + xinc, ye[k + eoffset] + yinc, mag[k + eoffset] + minc, i, j);
		for(k = index + 1; k <= N; k++){
			res += multiEvaluate(xe[k + eoffset], ye[k + eoffset], mag[k + eoffset], i, j);
		}
		float e = res-M.elements[offset + i];
            residuals += e * e;
        }
    }
	return residuals;
}

__device__ float multiEvaluate(float x0, float y0, float mag, int x, int y) {
	return mag * __expf(-__fdividef(((x - x0)*(x - x0) + (y - y0)*(y - y0)), (_2sig2)));
}

void checkCudaError(){
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf((stderr, "Runtime API error %d: %s.\n", (int)err, cudaGetErrorString( err ) ));
        exit(-1);
	}
	return;
}
