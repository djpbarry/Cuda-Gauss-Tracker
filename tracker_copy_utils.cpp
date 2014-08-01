
#include "stdafx.h"
#include <tracker_copy_utils.h>
#include <defs.h>

//Copy statevectors corresponding to frame index from source to dest

void copyStateVector(float* dest, float* source, int index, int _currentLength) {
    int boffset = index * MAX_DETECTIONS * DIM_OF_STATE;
    for (int i = 0; i < _currentLength; i++) {
        int soffset = i * DIM_OF_STATE;
        for (int j = 0; j < DIM_OF_STATE; j++) {
            dest[boffset + soffset + j] = source[soffset + j];
        }
    }
    return;
}

void copyStateParticles(float* dest, float* source, int stateVectorParticleIndex, int _mNbParticles) {
    for (int i = 0; i < _mNbParticles; i++) {
        int particleIndex = stateVectorParticleIndex + i * (DIM_OF_STATE + 1);
        for (int j = 0; j < DIM_OF_STATE + 1; j++) {
            dest[j + particleIndex - stateVectorParticleIndex] = source[j + particleIndex];
        }
    }
    return;
}

void copyStateParticlesToMemory(float* dest, float* source, int frameIndex, int _mNbParticles, int totalLength, int offset) {
    if (dest == NULL) return;
    int frameVectorIndex = frameIndex * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1);
    for (int k = offset; k < totalLength; k++) {
        int stateVectorIndex = frameVectorIndex + _mNbParticles * (DIM_OF_STATE + 1) * k;
        for (int i = 0; i < _mNbParticles; i++) {
            int particleIndex = stateVectorIndex + i * (DIM_OF_STATE + 1);
            for (int j = 0; j < DIM_OF_STATE + 1; j++) {
                dest[j + particleIndex] = source[j + particleIndex - frameVectorIndex];
            }
        }
    }
    return;
}

void copyParticle(float* dest, float* source, int index) {
    for (int i = 0; i < DIM_OF_STATE + 1; i++) {
        dest[i + index] = source[i];
    }
    return;
}

void updateStateVector(float* vector, int index, float* _mStateVectors) {
    int offset = index * DIM_OF_STATE;
    for (int i = 0; i < DIM_OF_STATE; i++) {
        _mStateVectors[i + offset] = vector[i];
    }
    return;
}
