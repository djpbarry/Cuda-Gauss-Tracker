#ifndef _TRACKER_COPY_UTILS
#define _TRACKER_COPY_UTILS


extern void copyStateVector(float* dest, float* source, int index, int _currentLength);
extern void copyStateParticles(float* dest, float* source, int stateVectorParticleIndex, int _mNbParticles);
extern void copyStateParticlesToMemory(float* dest, float* source, int frameIndex, int _mNbParticles, int totalLength, int offset);
extern void copyParticle(float* dest, float* source, int index);

#endif