#ifndef _TRACKER_UTILS_
#define _TRACKER_UTILS_

#include "stdafx.h"
#include <defs.h>
#include <matrix_mat.h>
#include <boost/math/special_functions/round.hpp>
#include <boost/lexical_cast.hpp>

using namespace boost;

extern void output(int* dims, int frames, string outputDir, int _mNbParticles, int* _counts, float* _mParticlesMemory, int _scalefactor);
extern void copyStateVector(float* dest, float* source, int index, int _currentLength);
extern void copyStateParticles(float* dest, float* source, int stateVectorParticleIndex, int _mNbParticles);
extern void copyStateParticlesToMemory(float* dest, float* source, int frameIndex, int _mNbParticles, int _currentLength);
extern void copyParticle(float* dest, float* source, int index);

#endif