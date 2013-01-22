#ifndef _CUDA_TRACKER_
#define _CUDA_TRACKER_

extern "C" void updateParticleWeightsOnGPU(Matrix observation, float* mParticles, int totalLength, int nbParticles, int offset);

#endif