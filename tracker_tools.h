#ifndef _TRACKER_TOOLS_
#define _TRACKER_TOOLS_

#include <matrix.h>

extern void createParticles(float* aStateVectors, float* aParticles, int totalLength, int _mNbParticles, int offset);
extern void filterTheInitialization(Matrix aImageStack, int aInitPFIterations, float* _mParticles, int _currentLength, int _mNbParticles, float* _mStateVectors, int offset);
extern void scaleSigmaOfRW(float vScaler);
extern void DrawParticlesWithRW(float* aParticles, int totalLength, int _mNbParticles, int offset);
extern void randomWalkProposal(float* aParticles, int offset);
//extern void updateParticleWeights(Matrix aObservationStack, int aFrameIndex, int _currentLength, int _mNbParticles, float* _mParticles, float* _mMaxLogLikelihood);
extern void generateParticlesIntensityBitmap(float* setOfParticles, int stateVectorIndex, bool* vBitmap, int width, int height, int _mNbParticles);
extern void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int stateVectorIndex, int _mNbParticles);
extern void generateIdealImage(float* particles, int offset, float* vIdealImage, int width, int height);
extern void addBackgroundToImage(float* aImage, float aBackground, int width, int height);
extern void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity, int width, int height);
extern float calculateLogLikelihood(Matrix aStackProcs, int aFrame, float* aGivenImage, bool* aBitmap);
extern bool runParticleFilter(Matrix aOriginalImage, float* _mParticles, float* _mParticlesMemory, float* _mStateVectors, float* _mStateVectorsMemory, int* _counts, int _currentLength, int _mNbParticles, int _mInitRWIterations);
extern void drawNewParticles(float* aParticlesToRedraw, float spatialRes, int _currentLength, int _mNbParticles);
extern void drawFromProposalDistribution(float* particles, float spatialRes, int particleIndex);
extern int checkStateVectors(float* stateVectors, float* particles, int width, int height, int nVectors, int nParticles);
extern void updateStateVector(float* vector, int index, float* _mStateVectors);
extern void estimateStateVectors(float* aStateVectors, float* aParticles, int totalLength, int _mNbParticles, int offset);
extern bool resample(float* aParticles, int totalLength, int _mNbParticles, int offset);

#endif