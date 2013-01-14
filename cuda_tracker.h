#ifndef _CUDA_TRACKER_
#define _CUDA_TRACKER_

extern "C" void updateParticleWeightsOnGPU(Matrix observation, float* mParticles, int currentLength, int nbParticles);

#endif