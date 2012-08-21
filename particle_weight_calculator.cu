
#include "stdafx.h"
#include <matrix.h>
#include <math.h>
#include <float.h>
#include <defs.h>
#include <cuda_utils.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

extern "C" void updateParticleWeightsOnGPU(Matrix observation, float* mParticles, int currentLength, int nbParticles);
__global__ void LogLikelihoodKernel(Matrix observation, float* mParticles, float* vIdealImage, float* logLikelihoods, float vVarianceXYinPx);
__device__ float calculateLogLikelihood(Matrix aStackProcs, float* aGivenImage);
__device__ void generateIdealImage(float* particles, int offset, float* vIdealImage, int width, int height, float vVarianceXYinPx);
__device__ void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity, int width, int height, float vVarianceXYinPx);

extern "C" float _mSigmaPSFxy, _spatialRes;

/*
* Parallelised updating of particle weights in particle filter for tracking application. Weights of all particles for all state vectors (objects)
* for a given image frame are calculated simultaneously. The total number of threads to be executed on the GPU is equal to
* <code>currentLength * nbParticles</code>.
*
* Required Inputs:
* @param observation the observed image
* @param frameIndex the index of the current observed image within the observed sequence
* @param currentLength the current number of objects being tracked
* @param nbParticles the number of particles for each object
*/
extern "C" void updateParticleWeightsOnGPU(Matrix observation, float* mParticles, int currentLength, int nbParticles){
	float vVarianceXYinPx = _mSigmaPSFxy * _mSigmaPSFxy / (_spatialRes * _spatialRes);
	cudaSetDevice(0);
	checkCudaError();
    
	// Observed image on GPU device
	Matrix d_observation; 
    d_observation.width = observation.width;
	d_observation.height = observation.height; 
	d_observation.stride = observation.stride;
	d_observation.size = d_observation.width * d_observation.height;
    size_t observationSize = observation.width * observation.height * sizeof(float);
	cudaMalloc(&d_observation.elements, observationSize);
	checkCudaError();
	cudaMemcpy(d_observation.elements, observation.elements, observationSize, cudaMemcpyHostToDevice);
	checkCudaError();

	// Particles on GPU device
	float* d_mParticles;
	size_t mParticlesSize = currentLength * nbParticles * (DIM_OF_STATE + 1) * sizeof(float);
	cudaMalloc(&d_mParticles, mParticlesSize);
	checkCudaError();
	cudaMemcpy(d_mParticles, mParticles, mParticlesSize, cudaMemcpyHostToDevice);
	checkCudaError();

	// Ideal images on GPU device
	float* d_vIdealImage;
	size_t idealImagesSize = currentLength * nbParticles * observation.width * observation.height * sizeof(float);
	cudaMalloc(&d_vIdealImage, idealImagesSize);
	checkCudaError();

	// Log likelihoods on GPU device
	float* d_vLogLikelihoods;
	size_t logLikelihoodsSize = currentLength * nbParticles * sizeof(float);
	cudaMalloc(&d_vLogLikelihoods, logLikelihoodsSize);
	checkCudaError();

	int totalParticles = currentLength * nbParticles;
	int numblocks = totalParticles / BLOCK_SIZE_X;
	if (numblocks * BLOCK_SIZE_X < totalParticles) numblocks++;
	if (numblocks % 2 != 0) numblocks++;

    dim3 dimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 dimGrid(numblocks, 1);

	LogLikelihoodKernel<<<dimGrid, dimBlock>>>(d_observation, d_mParticles, d_vIdealImage, d_vLogLikelihoods, vVarianceXYinPx);

	cudaMemcpy(mParticles, d_mParticles, mParticlesSize, cudaMemcpyDeviceToHost);
	checkCudaError();

	float* vLogLikelihoods = (float*)malloc(sizeof(float) * nbParticles * currentLength);

	cudaMemcpy(vLogLikelihoods, d_vLogLikelihoods, logLikelihoodsSize, cudaMemcpyDeviceToHost);
	checkCudaError();

	cudaFree(d_mParticles);
	checkCudaError();
	cudaFree(d_vLogLikelihoods);
	checkCudaError();
	cudaFree(d_vIdealImage);
	checkCudaError();

	cudaDeviceReset();
	checkCudaError();

	for(int i=0; i < currentLength; i++){
		int logStateIndex = i * nbParticles;
		int stateVectorIndex = i * nbParticles * (DIM_OF_STATE + 1);
        float vSumOfWeights = 0.0f;
        //
        // Calculate the likelihoods for each particle and save the biggest one
        //
        float vMaxLogLikelihood = -FLT_MAX;
        for (int vI = 0; vI < nbParticles; vI++) {
            if (vLogLikelihoods[logStateIndex + vI] > vMaxLogLikelihood) {
                vMaxLogLikelihood = vLogLikelihoods[logStateIndex + vI];
            }
        }
        //_mMaxLogLikelihood[aFrameIndex - 1] = vMaxLogLikelihood;
        //
        // Iterate again and update the weights
        //
		for (int vI = 0; vI < nbParticles; vI++) {
            vLogLikelihoods[logStateIndex + vI] -= vMaxLogLikelihood;
			int particleWeightIndex = stateVectorIndex + vI * (DIM_OF_STATE + 1) + DIM_OF_STATE;
            mParticles[particleWeightIndex] = mParticles[particleWeightIndex] * expf(vLogLikelihoods[logStateIndex + vI]);
            vSumOfWeights += mParticles[particleWeightIndex];
        }
        //
        // Iterate again and normalize the weights
        //
        if (vSumOfWeights == 0.0f) { //can happen if the winning particle before had a weight of 0.0
			for (int vI = 0; vI < nbParticles; vI++) {
                mParticles[stateVectorIndex + vI * (DIM_OF_STATE+1) + DIM_OF_STATE] = 1.0f / (float)nbParticles;
            }
        } else {
			for (int vI = 0; vI < nbParticles; vI++) {
                mParticles[stateVectorIndex + vI * (DIM_OF_STATE+1) + DIM_OF_STATE] /= vSumOfWeights;
            }
        }
    }
	free(vLogLikelihoods);
	return;
}

__global__ void LogLikelihoodKernel(Matrix observation, float* mParticles, float* vIdealImage, float* logLikelihoods, float vVarianceXYinPx){
	//calculate ideal image
	int particleIndex =  blockIdx.x * blockDim.x + threadIdx.x;
	generateIdealImage(mParticles, particleIndex, vIdealImage, observation.width, observation.height, vVarianceXYinPx);
	//calculate likelihood
	logLikelihoods[particleIndex] = calculateLogLikelihood(observation, vIdealImage);
	return;
}

__device__ float calculateLogLikelihood(Matrix aStackProcs, float* aGivenImage){
    float vLogLikelihood = 0;
	for (int vY = 0; vY < aStackProcs.height; vY++) {
		int woffset = vY * aStackProcs.width;
		for (int vX = 0; vX < aStackProcs.width; vX++) {
			vLogLikelihood += -aGivenImage[woffset + vX] + aStackProcs.elements[woffset + vX] * log(aGivenImage[woffset + vX]);
        }
    }
    return vLogLikelihood;
}

__device__ void generateIdealImage(float* particles, int offset, float* vIdealImage, int width, int height, float vVarianceXYinPx) {
    addFeaturePointToImage(vIdealImage, particles[offset], particles[offset + 1], particles[offset + 6], width, height, vVarianceXYinPx);
    return;
}

__device__ void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity, int width, int height, float vVarianceXYinPx) {
    for (int vY = 0; vY < height; vY++) {
		int woffset = vY * width;
        for (int vX = 0; vX <width; vX++) {
            aImage[woffset + vX] += (aIntensity * expf(-(powf(vX - x + .5f, 2.0f) + powf(vY - y + .5f, 2.0f)) / (2.0f * vVarianceXYinPx)));
        }
    }
}