#include "stdafx.h"
#include <matrix.h>
#include <utils.h>
#include <defs.h>
#include <vector>
#include <iterator>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/math/special_functions/round.hpp>

using namespace boost;

extern "C" void updateParticleWeightsOnGPU(Matrix observation, float* mParticles, int currentLength, int nbParticles);
int initTracker(char* ext);
void setup();
void runTracker();
void initParticleFilter(Matrix aInitStack, int aInitParticleFilterIterations, int aFrameOfInit);
void createParticles(float* aStateVectors, float* aParticles);
void filterTheInitialization(Matrix aImageStack, int aInitPFIterations, int aFrameOfInit);
void scaleSigmaOfRW(float vScaler);
void DrawParticlesWithRW(float* aParticles);
void randomWalkProposal(float* aParticles, int offset);
void updateParticleWeights(Matrix aObservationStack, int aFrameIndex);
void generateParticlesIntensityBitmap(float* setOfParticles, int offset, bool* vBitmap);
void estimateStateVectors(float* aStateVectors, float* aParticles);
bool resample(float* aParticles);
void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int offset);
void generateIdealImage(float* particles, int offset, float* vIdealImage);
void addBackgroundToImage(float* aImage, float aBackground);
void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity, int width, int height);
float calculateLogLikelihood(Matrix aStackProcs, int aFrame, float* aGivenImage, bool* aBitmap);
void runParticleFilter(Matrix aOriginalImage);
void drawNewParticles(float* aParticlesToRedraw, float spatialRes);
void drawFromProposalDistribution(float* particles, float spatialRes, int particleIndex);
void copyStateVector(float* dest, float* source, int index);
void copyStateParticles(float* dest, float* source, int stateVectorParticleIndex);
void copyParticle(float* dest, float* source, int index);
void updateStateVector(float* vector, int index);
extern "C" int maxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z);

extern "C" float _spatialRes;
extern "C" int _scalefactor;
int _mHeight;
int _mWidth;
int _mNFrames;
int _mTrackTillFrameNb;
float* _mStateVectors;
float* _mParticles;
float* _mStateVectorsMemory;
float* _mMaxLogLikelihood;
int _mFrameOfInitialization = 0;
int _mInitRWIterations = 1;
float _mWavelengthInNm = 650.0f;
float _mNA = 1.4f;
extern "C" float _mSigmaPSFxy = (0.21f * _mWavelengthInNm / _mNA);
float _mSigmaOfRandomWalk[] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
float _mBackground = 1.0f;
float _mSigmaOfDynamics[] = {200.0f, 200.0f, 1.0f, 1.0f};
bool _mDoResampling = true;
bool _mDoPrecisionOptimization = true;
int _currentLength;
int _mNbParticles = 100;
int _mResamplingThreshold = _mNbParticles / 2;
int _mRepSteps = 1;
normal_distribution<float> _dist(0.0f, 1.0f);
mt19937 rng;
variate_generator<mt19937, normal_distribution<float> > var_nor(rng, _dist);
Matrix _mOriginalImage;
char* folder = "C:/Users/barry05/Desktop/Tracking Test Sequences/TiffSim3";

int main(int argc, char* argv[]){
	int frames = initTracker(".tif");
	runTracker();
	printf("\n\n");

	return 0;
}

int initTracker(char* ext){
	printf("Start Tracker...\n\nFolder: %s\n", folder);
	
	//Load file list
	vector<path> v = getFiles(folder);

	//Count files with specified extension
	int numFiles = countFiles(v, ext);
	vector<path>::iterator v_iter;
	
	//Get dimensions of first image
	int dims[2];
	getDims(v, ext, dims);

	//Construct image volume
	_mOriginalImage.width = dims[0];
	_mOriginalImage.stride = dims[0];
	_mOriginalImage.height = dims[1];
	_mOriginalImage.depth = numFiles;
	_mOriginalImage.size = dims[0] * dims[1] * numFiles;
	_mOriginalImage.elements = (float*)malloc(sizeof(float) * _mOriginalImage.size);

	//Load images into volume
	printf("\nLoading Images ... %d%%", 0);
	int thisFrame = 0;
	Mat frame;
	for(v_iter = v.begin(); v_iter != v.end(); v_iter++){
		string ext_s = ((*v_iter).extension()).string();
		if((strcmp(ext_s.c_str(), ext) == 0)) {
			printf("\rLoading Images ... %d%%", ((thisFrame + 1) * 100) / numFiles);
			frame = imread((*v_iter).string(), -1);
			copyToMatrix(frame, _mOriginalImage, thisFrame);
			thisFrame++;
		}
	}
	setup();
	return thisFrame;
}

//Initlialise variables
void setup() {
    _mHeight = _mOriginalImage.height;
    _mWidth = _mOriginalImage.width;
	_mNFrames = _mOriginalImage.depth;
    _mTrackTillFrameNb = _mNFrames;
    _mStateVectorsMemory = (float*)malloc(sizeof(float) * _mNFrames * MAX_DETECTIONS * DIM_OF_STATE);
	_mStateVectors = (float*)malloc(sizeof(float) * MAX_DETECTIONS * DIM_OF_STATE);
    _mMaxLogLikelihood = (float*)malloc(sizeof(float) * _mNFrames);
    return;
}

void runTracker(){
	// Storage for regions containing candidate particles
	Matrix candidates;

	// Find local maxima and use to initialise state vectors
	candidates.width = FIT_SIZE * MAX_DETECTIONS;
	candidates.stride = candidates.width;
	candidates.height = FIT_SIZE + DATA_ROWS;
	candidates.size = candidates.width * candidates.height;
	candidates.elements = (float*)malloc(sizeof(float) * candidates.size);
	_currentLength = 0;
	_currentLength = maxFinder(_mOriginalImage, candidates, 1.0f, false, _currentLength, 0, 0);
	for (int i = 0; i < _currentLength; i++) {
		float x = candidates.elements[i];
		float y = candidates.elements[i + candidates.stride];
		float mag = candidates.elements[i * FIT_SIZE + FIT_RADIUS + candidates.stride * (HEADER + FIT_RADIUS)];
        float firstState[] = {x, y, 0.0f, 0.0f, 0.0f, 0.0f, mag};
		updateStateVector(firstState, i);
	}
	free(candidates.elements);
	_mParticles = (float*)malloc(sizeof(float) * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1));
	initParticleFilter(_mOriginalImage, _mInitRWIterations, _mFrameOfInitialization);
	copyStateVector(_mStateVectorsMemory, _mStateVectors, 0);
	runParticleFilter(_mOriginalImage);
}

void initParticleFilter(Matrix aInitStack, int aInitParticleFilterIterations, int aFrameOfInit) {
        // - set up state vector
        // - create particles
        // - filter the initialized values	
    createParticles(_mStateVectors, _mParticles);
    filterTheInitialization(aInitStack, aInitParticleFilterIterations, aFrameOfInit);
	return;
}

void createParticles(float* aStateVectors, float* aParticles){
	printf("\nCreating Particles ... %d%%", 0);
    for (int i=0; i < _currentLength; i++) {
		printf("\rCreating Particles ... %d%%", ((i + 1) * 100) / _currentLength);
		int stateVectorIndex = i * DIM_OF_STATE;
		int stateVectorParticleIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        for (int vIndex = 0; vIndex < _mNbParticles; vIndex++) {
            float vProposal[DIM_OF_STATE + 1];
			int particleIndex = stateVectorParticleIndex + vIndex * (DIM_OF_STATE + 1);
            for (int vI = 0; vI < DIM_OF_STATE; vI++) {
                vProposal[vI] = aStateVectors[stateVectorIndex + vI];
            }
//				Init the weight as a last dimension
            vProposal[DIM_OF_STATE] = 1.0f; //not 0!			
            //add the new particle
            copyParticle(aParticles, vProposal, particleIndex);
        }
    }
	return;
}

void filterTheInitialization(Matrix aImageStack, int aInitPFIterations, int aFrameOfInit){
	float vSigmaOfRWSave[DIM_OF_STATE];
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        vSigmaOfRWSave[vI] = _mSigmaOfRandomWalk[vI];
    }
	Matrix frame;
	frame.width = aImageStack.width;
	frame.height = aImageStack.height;
	frame.stride = frame.width;
	frame.depth = 1;
	frame.size = frame.width * frame.height;
	frame.elements = (float*)malloc(sizeof(float) * frame.size);
	matrixCopy(aImageStack, frame, 0);
 	
    for (int vR = 0; vR < aInitPFIterations; vR++) {
        scaleSigmaOfRW(1.0f / powf(3.0f, (float)vR));
        DrawParticlesWithRW(_mParticles);

		//updateParticleWeightsOnGPU(frame, _mParticles, _currentLength, _mNbParticles);
        updateParticleWeights(aImageStack, aFrameOfInit);

        estimateStateVectors(_mStateVectors, _mParticles);

        resample(_mParticles);
    }

    //restore the sigma vector
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        _mSigmaOfRandomWalk[vI] = vSigmaOfRWSave[vI];
    }
	free(frame.elements);
	return;
}

void scaleSigmaOfRW(float vScaler) {
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        _mSigmaOfRandomWalk[vI] *= vScaler;
    }
}

void DrawParticlesWithRW(float* aParticles) {
	//printf("\nDrawing Particles ... %d%%", 0);
	for(int i=0; i<_currentLength;i++){
		//printf("\rDrawing Particles ... %d%%", ((i + 1) * 100) / _currentLength);
		int stateVectorIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        for (int j=0; j<_mNbParticles; j++){
			int particleIndex = stateVectorIndex + j * (DIM_OF_STATE + 1);
			randomWalkProposal(aParticles, particleIndex);
        }
	}
    return;
}

void randomWalkProposal(float* aParticles, int offset) {
    for (int aI = 0; aI < DIM_OF_STATE; aI++) {
        aParticles[aI + offset] += var_nor() * _mSigmaOfRandomWalk[aI];
    }
	return;
}

void updateParticleWeights(Matrix aObservationStack, int aFrameIndex){
	//printf("\nUpdating Weights ... %d%%", 0);
	for(int i=0; i<_currentLength; i++){
		//printf("\rUpdating Weights ... %d%%", ((i + 1) * 100) / _currentLength);
		int stateVectorIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        float vSumOfWeights = 0.0f;
        //
        // Calculate the likelihoods for each particle and save the biggest one
        //
        float* vLogLikelihoods = (float*)malloc(sizeof(float) * _mNbParticles);
        float vMaxLogLikelihood = -FLT_MAX;
		bool* vBitmap = (bool*)malloc(sizeof(bool) * _mWidth * _mHeight);
        generateParticlesIntensityBitmap(_mParticles, stateVectorIndex, vBitmap);
        calculateLikelihoods(aObservationStack, vBitmap, aFrameIndex, vLogLikelihoods, _mParticles, stateVectorIndex);
        for (int vI = 0; vI < _mNbParticles; vI++) {
            if (vLogLikelihoods[vI] > vMaxLogLikelihood) {
                vMaxLogLikelihood = vLogLikelihoods[vI];
            }
        }
        _mMaxLogLikelihood[aFrameIndex - 1] = vMaxLogLikelihood;
        //
        // Iterate again and update the weights
        //
        int vI = 0;
		for (int vI = 0; vI < _mNbParticles; vI++) {
            vLogLikelihoods[vI] -= vMaxLogLikelihood;
			int particleWeightIndex = stateVectorIndex + vI * (DIM_OF_STATE + 1) + DIM_OF_STATE;
            _mParticles[particleWeightIndex] = _mParticles[particleWeightIndex] * expf(vLogLikelihoods[vI]);
            vSumOfWeights += _mParticles[particleWeightIndex];
        }
        //
        // Iterate again and normalize the weights
        //
        if (vSumOfWeights == 0.0f) { //can happen if the winning particle before had a weight of 0.0
			for (int vI = 0; vI < _mNbParticles; vI++) {
                _mParticles[stateVectorIndex + vI * (DIM_OF_STATE+1) + DIM_OF_STATE] = 1.0f / (float)_mNbParticles;
            }
        } else {
			for (int vI = 0; vI < _mNbParticles; vI++) {
                _mParticles[stateVectorIndex + vI * (DIM_OF_STATE+1) + DIM_OF_STATE] /= vSumOfWeights;
            }
        }
		free(vLogLikelihoods);
		free(vBitmap);
    }
	return;
}

//Estimates all state vectors from the particles and their weights
void estimateStateVectors(float* aStateVectors, float* aParticles){
	//printf("\nEstimating Vectors ... %d%%", 0);
	for(int i=0; i< _currentLength; i++){
		//printf("\rEstimating Vectors ... %d%%", ((i+1) * 100)/_currentLength);
		int stateVectorIndex = i * DIM_OF_STATE;
		int stateVectorParticleIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        /*
            * Set the old state to 0
            */
        for (int vI = 0; vI < DIM_OF_STATE; vI++) {
            aStateVectors[stateVectorIndex + vI] = 0.0f;
        }

        for (int pI = 0; pI < _mNbParticles; pI++) {
			int particleIndex = stateVectorParticleIndex + pI * (DIM_OF_STATE + 1);
            for (int vDim = 0; vDim < DIM_OF_STATE; vDim++) {
                aStateVectors[stateVectorIndex + vDim] += aParticles[particleIndex + DIM_OF_STATE] * aParticles[particleIndex + vDim];
            }
        }
    }
	return;
}

	/**
     *
     * @param aParticles set of parameters to resample.
     * @return true if resampling was performed, false if not.
     */
bool resample(float* aParticles){
	//printf("\nResampling ... %d%%", 0);
	for(int i=0; i < _currentLength; i++){
		//printf("\rResampling ... %d%%", ((i+1) * 100)/_currentLength);
		int stateVectorParticleIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        //
        // First check if the threshold is smaller than Neff
        //
        float vNeff = 0;
		for(int j=0; j<_mNbParticles; j++){
			int particleIndex = stateVectorParticleIndex + j * (DIM_OF_STATE + 1);
            vNeff += aParticles[particleIndex + DIM_OF_STATE] * aParticles[particleIndex + DIM_OF_STATE];
        }
        vNeff = 1.0f / vNeff;

        if (vNeff > _mResamplingThreshold) {
//				System.out.println("no resampling");
            return false; //we won't do the resampling
        }
        //
        // Begin resampling
        //
//			System.out.println("Resampling");
        float VNBPARTICLES_1 = 1.0f / (float) _mNbParticles;
        double* vC = (double*)malloc(sizeof(double) * (_mNbParticles + 1));
        vC[0] = 0.0;
        for (int vInd = 1; vInd <= _mNbParticles; vInd++) {
			int particleIndex = stateVectorParticleIndex + (vInd - 1) * (DIM_OF_STATE + 1);
            vC[vInd] = vC[vInd - 1] + aParticles[particleIndex + DIM_OF_STATE];
        }

        double vU = (rand() * (double)VNBPARTICLES_1)/RAND_MAX;

        float* vFPParticlesCopy = (float*)malloc(sizeof(float) * _mNbParticles * (DIM_OF_STATE + 1));
		copyStateParticles(vFPParticlesCopy, aParticles, stateVectorParticleIndex);
		int vI = 0;
        for (int vParticleCounter = 0; vParticleCounter < _mNbParticles; vParticleCounter++) {
            while (vU > vC[vI]) {
                if (vI < _mNbParticles) //this can happen due to numerical reasons
                {
                    vI++;
                }
            }
			int sourceParticleIndex = (vI - 1) * (DIM_OF_STATE + 1);
			int destParticleIndex = stateVectorParticleIndex + vParticleCounter * (DIM_OF_STATE + 1);
			//int particleIndex = stateVectorParticleIndex + (vI - 1) * (DIM_OF_STATE + 1);
           for (int vK = 0; vK < DIM_OF_STATE; vK++) {
                aParticles[destParticleIndex + vK] = vFPParticlesCopy[sourceParticleIndex + vK];
            }
            aParticles[destParticleIndex + DIM_OF_STATE] = VNBPARTICLES_1;
            vU += VNBPARTICLES_1;
        }
		free(vC);
		free(vFPParticlesCopy);
    }
	return true;
}

void generateParticlesIntensityBitmap(float* setOfParticles, int stateVectorIndex, bool* vBitmap) {
// convert to pixel distance and multiply with 3: 4
    float vMaxDistancexy = (3.0f * _mSigmaPSFxy / _spatialRes);
// get a bounding box around the each feature point
    int vXStart, vXEnd, vYStart, vYEnd;
	//Initiliase array
	for (int vY = 0; vY < _mHeight; vY++) {
		int boffset = vY * _mWidth;
        for (int vX = 0; vX < _mWidth; vX++) {
            vBitmap[boffset + vX] = false;
        }
    }
	for( int i=0; i<_mNbParticles; i++){
		int particleIndex = stateVectorIndex + i * (DIM_OF_STATE + 1);
        if (setOfParticles[particleIndex] - vMaxDistancexy < 0) {
            vXStart = 0;
        } else {
			vXStart = (int)boost::math::round<float>(setOfParticles[particleIndex] - vMaxDistancexy);
        }
        if (setOfParticles[particleIndex] + vMaxDistancexy >= _mWidth) {
            vXEnd = _mWidth - 1;
        } else {
            vXEnd = (int)boost::math::round<float>(setOfParticles[particleIndex] + vMaxDistancexy);
        }
        if (setOfParticles[particleIndex + 1] - vMaxDistancexy < 0) {
            vYStart = 0;
        } else {
            vYStart = (int)boost::math::round<float>(setOfParticles[particleIndex + 1] - vMaxDistancexy);
        }
        if (setOfParticles[particleIndex + 1] + vMaxDistancexy >= _mHeight) {
            vYEnd = _mHeight - 1;
        } else {
            vYEnd = (int)boost::math::round<float>(setOfParticles[particleIndex + 1] + vMaxDistancexy);
        }
        for (int vY = vYStart; vY <= vYEnd && vY < _mHeight; vY++) {
			int boffset = vY * _mWidth;
            for (int vX = vXStart; vX <= vXEnd && vX < _mWidth; vX++) {
                vBitmap[boffset + vX] = true;
            }
        }
    }
    return;
}

void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int stateVectorIndex){
	//calculate ideal image
	for(int vI = 0; vI < _mNbParticles; vI++){
		int particleIndex = stateVectorIndex + vI * (DIM_OF_STATE + 1);
		float* vIdealImage = (float*)malloc(sizeof(float) * _mWidth * _mHeight);
		generateIdealImage(mParticles, particleIndex, vIdealImage);
		//calculate likelihood
		vLogLikelihoods[vI] = calculateLogLikelihood(aObservationStack, aFrameIndex, vIdealImage, vBitmap);
		free(vIdealImage);
	}
	return;
}

void generateIdealImage(float* particles, int offset, float* vIdealImage) {
    addBackgroundToImage(vIdealImage, _mBackground);
    addFeaturePointToImage(vIdealImage, particles[offset], particles[offset + 1], particles[offset + 6], _mWidth, _mHeight);
    return;
}

void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity, int width, int height) {
    float vVarianceXYinPx = _mSigmaPSFxy * _mSigmaPSFxy / (_spatialRes * _spatialRes);
    float vMaxDistancexy = (5.0f * _mSigmaPSFxy / _spatialRes);

    int vXStart, vXEnd, vYStart, vYEnd;//defines a bounding box around the tip
    if (x + .5f - (vMaxDistancexy + .5f) < 0) {
        vXStart = 0;
    } else {
        vXStart = (int) (x + .5f) - (int) (vMaxDistancexy + .5f);
    }
    if (y + .5f - (vMaxDistancexy + .5f) < 0) {
        vYStart = 0;
    } else {
        vYStart = (int) (y + .5f) - (int) (vMaxDistancexy + .5f);
    }
    if (x + .5f + (vMaxDistancexy + .5f) >= width) {
        vXEnd = width - 1;
    } else {
        vXEnd = (int) (x + .5f) + (int) (vMaxDistancexy + .5f);
    }
    if (y + .5f + (vMaxDistancexy + .5f) >= height) {
        vYEnd = height - 1;
    } else {
        vYEnd = (int) (y + .5f) + (int) (vMaxDistancexy + .5f);
    }
    for (int vY = vYStart; vY <= vYEnd && vY < height; vY++) {
		int woffset = vY * width;
        for (int vX = vXStart; vX <= vXEnd && vX <width; vX++) {
            aImage[woffset + vX] += (aIntensity * expf(-(powf(vX - x + .5f, 2.0f) + powf(vY - y + .5f, 2.0f)) / (2.0f * vVarianceXYinPx)));
        }
    }
}

void addBackgroundToImage(float* aImage, float aBackground) {
	for(int y=0; y < _mHeight; y++){
		int offset = y * _mWidth;
		for(int x=0; x < _mWidth; x++){
			aImage[offset + x] = aBackground;
		}
	}
	return;
}

/**
     * Calculates the likelihood by multipling the poissons marginals around a
     * particle given a image(optimal image)
     *
     * @param aImagePlus: The observed image
     * @param aFrame: The frame index 1<=n<=NSlices(to read out the correct
     * substack from aStack
     * @param aGivenImage: a intensity array, the 'measurement'
     * @param aBitmap: Pixels which are not set to true in the bitmap are pulled
     * out(proportionality)
     * @return the likelihood for the image given
     */
float calculateLogLikelihood(Matrix aStackProcs, int aFrame, float* aGivenImage, bool* aBitmap) //ImageStack aImageStack){
{
    float vLogLikelihood = 0;
    //we need all processors anyway. Profiling showed that the method getProcessor needs a lot of time. Store them
    //in an Array.
	int foffset = aFrame * _mWidth * _mHeight;
//		long vTime1 = System.currentTimeMillis();
        for (int vY = 0; vY < _mHeight; vY++) {
			int woffset = vY * _mWidth;
            for (int vX = 0; vX < _mWidth; vX++) {
                if (aBitmap[woffset + vX]) {
					float a = -aGivenImage[woffset + vX];
					float b = aStackProcs.elements[foffset + woffset + vX];
					float c = log(aGivenImage[woffset + vX]);
					vLogLikelihood += a + b * c;
                }
            }
        }
//		System.out.println("used time for loglik = " + (System.currentTimeMillis() - vTime1));
    //IJ.showStatus("likelihood finshed");
    return vLogLikelihood;
}

void runParticleFilter(Matrix aOriginalImage) {
	Matrix output;
	output.width = _mWidth*_scalefactor;
	output.stride=output.width;
	output.height = _mHeight*_scalefactor;
	output.size = output.width * output.height;
	output.elements = (float*)malloc(sizeof(float) * output.size);
	printf("\nRunning Particle Filter ... %d%%", 0);
	Matrix frame;
	frame.width = aOriginalImage.width;
	frame.height = aOriginalImage.height;
	frame.stride = frame.width;
	frame.depth = 1;
	frame.size = frame.width * frame.height;
	frame.elements = (float*)malloc(sizeof(float) * frame.size);
			
    for (int vFrameIndex = _mFrameOfInitialization; vFrameIndex < _mTrackTillFrameNb; vFrameIndex++) {
		printf("\rRunning Particle Filter ... %d%%", (vFrameIndex + 1) * 100 / _mTrackTillFrameNb);
		matrixCopy(aOriginalImage, frame, vFrameIndex * aOriginalImage.width * aOriginalImage.height);
		// save a copy of the original sigma to restore it afterwards
		float vSigmaOfRWSave[DIM_OF_STATE];
		for (int vI = 0; vI < DIM_OF_STATE; vI++) {
			vSigmaOfRWSave[vI] = _mSigmaOfRandomWalk[vI];
		}
		for (int vRepStep = 0; vRepStep < _mRepSteps; vRepStep++) {
			if (vRepStep == 0) {
				drawNewParticles(_mParticles, _spatialRes); //draw the particles at the appropriate position.
							_spatialRes /= _scalefactor;
			//update the view
			for(int p=0; p<output.width*output.height; p++){
				output.elements[p] = 0.0f;
			}
			Mat cudasaveframe(_mHeight*_scalefactor, _mWidth*_scalefactor,CV_32F);
			for(int i=0; i<_currentLength; i++){
				int stateVectorIndex = i * DIM_OF_STATE;
				/*addFeaturePointToImage(output.elements, _mStateVectors[stateVectorIndex] * _scalefactor,
					_mStateVectors[stateVectorIndex + 1] * _scalefactor,
					_mStateVectors[stateVectorIndex + 6], output.width, output.height);*/
				for(int j=0; j<_mNbParticles; j++){
					int particleIndex = (i * _mNbParticles + j) * (DIM_OF_STATE + 1);
					int x = (int)boost::math::round<float>(_mParticles[particleIndex] * _scalefactor);
					int y = (int)boost::math::round<float>(_mParticles[particleIndex + 1] * _scalefactor);
					addFeaturePointToImage(output.elements, x, y, _mParticles[particleIndex + DIM_OF_STATE], output.width, output.height);
					/*if(x > 0 && x < output.width && y > 0 && y < output.height){
						int index = x + y * output.stride;
						output.elements[index] += 1.0f;
					}*/
				}
			}
			copyFromMatrix(cudasaveframe, output, 0, 65535.0f/_mNbParticles);
			cudasaveframe.convertTo(cudasaveframe,CV_16UC1);
			string savefilename(folder);
			savefilename.append("/CudaOutput/");
			savefilename.append(boost::lexical_cast<string>(vFrameIndex));
			savefilename.append(".tif");
			imwrite(savefilename, cudasaveframe);
			_spatialRes *= _scalefactor;
			} else {
				scaleSigmaOfRW(1.0f / powf(3.0f, (float)vRepStep));//(1f - (float)vRepStep / (float)mRepSteps);
				DrawParticlesWithRW(_mParticles);
			}

			//updateParticleWeightsOnGPU(frame, _mParticles, _currentLength, _mNbParticles);
			updateParticleWeights(aOriginalImage, vFrameIndex);

			estimateStateVectors(_mStateVectors, _mParticles);

			if (_mDoResampling) {
				if (!resample(_mParticles)) {//further iterations are not necessary.
	//						System.out.println("number of iterations needed at this frame: " + vRepStep);
					break;
				}
			}
			if (!_mDoPrecisionOptimization) {
				break; //do not repeat the filter on the first frame
			}
		}
		//restore the sigma vector
		for (int vI = 0; vI < DIM_OF_STATE; vI++) {
			_mSigmaOfRandomWalk[vI] = vSigmaOfRWSave[vI];
		}
		//save the new states
		copyStateVector(_mStateVectorsMemory, _mStateVectors, vFrameIndex);
    }
	free(frame.elements);
}

/**
* Draws new particles for all the objects
*
*/
void drawNewParticles(float* aParticlesToRedraw, float spatialRes) {
	//invoke this method here to not repeat it for every particle, pass it by argument
	//TODO: Better would probably be to scale the sigma vector from beginning; this would then introduce the problem that 
	//		the sampling of the particles cannot be treated for each dimension independently, which introduces
	//		an error.
	for (int i=0; i<_currentLength; i++) {
		int stateVectorParticleIndex = i * (DIM_OF_STATE + 1) * _mNbParticles;
		for (int j=0; j<_mNbParticles; j++) {
			int particleIndex = stateVectorParticleIndex + j * (DIM_OF_STATE + 1);
			drawFromProposalDistribution(aParticlesToRedraw, spatialRes, particleIndex);
		}
	}
}

void drawFromProposalDistribution(float* particles, float spatialRes, int particleIndex) {
    particles[3 + particleIndex] = particles[3 + particleIndex] + var_nor() * (_mSigmaOfDynamics[0] / spatialRes);
    particles[4 + particleIndex] = particles[4 + particleIndex] + var_nor() * (_mSigmaOfDynamics[1] / spatialRes);
    particles[5 + particleIndex] = particles[5 + particleIndex] + var_nor() * (_mSigmaOfDynamics[2] / spatialRes);
    particles[particleIndex] = particles[particleIndex] + particles[3 + particleIndex];
    particles[1 + particleIndex] = particles[1 + particleIndex] + particles[4 + particleIndex];
    particles[2 + particleIndex] = particles[2 + particleIndex] + particles[5 + particleIndex];
    particles[6 + particleIndex] = particles[6 + particleIndex] + var_nor() * _mSigmaOfDynamics[3];
    if (particles[6 + particleIndex] < _mBackground + 1) {
        particles[6 + particleIndex] = _mBackground + 1;
    }
}

//Copy statevectors corresponding to frame index from source to dest
void copyStateVector(float* dest, float* source, int index){
	int boffset = index * MAX_DETECTIONS * DIM_OF_STATE;
	for(int i=0; i < _currentLength; i++){
		 int soffset = i * DIM_OF_STATE;
		 for(int j=0; j<DIM_OF_STATE; j++){
			 dest[boffset + soffset + j] = source[soffset + j];
		 }
	}
	return;
}

void copyStateParticles(float* dest, float* source, int stateVectorParticleIndex){
	for(int i=0; i < _mNbParticles; i++){
		int particleIndex = stateVectorParticleIndex + i * (DIM_OF_STATE + 1);
		for(int j=0; j < DIM_OF_STATE + 1; j++){
			dest[j + particleIndex - stateVectorParticleIndex] = source[j + particleIndex];
		}
	}
	return;
}

void copyParticle(float* dest, float* source, int index){
	for(int i=0; i < DIM_OF_STATE + 1; i++){
		 dest[i + index] = source[i];
	}
	return;
}

void updateStateVector(float* vector, int index){
	int offset = index * DIM_OF_STATE;
	for(int i=0; i < DIM_OF_STATE; i++){
		_mStateVectors[i + offset] = vector[i];
	}
	return;
}