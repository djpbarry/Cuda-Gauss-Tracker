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

void initTracker(char* ext);
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
void resample(float* aParticles);
void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int offset);
void generateIdealImage(float* particles, int offset, float* vIdealImage);
void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int offset);
void addBackgroundToImage(float* aImage, float aBackground);
void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity);
float calculateLogLikelihood(Matrix aStackProcs, int aFrame, float* aGivenImage, bool* aBitmap);
void copyStateVector(float* dest, float* source, int index);
void copyParticle(float* dest, float* source, int index);
void updateStateVector(float* vector, int index);
extern int maxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z);

extern float _spatialRes;
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
float _mSigmaPSFxy;
float _mWavelengthInNm = 650.0f;
float _mNA = 1.4f;
float _mSigmaOfRandomWalk[] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
float _mBackground = 1.0f;
float _mSigmaOfDynamics[] = {20.0f, 20.0f, 20.0f, 1.0f};
bool _mDoPrecisionCorrection = true;
int _currentLength;
int _mNbParticles = 10;
normal_distribution<float> _dist(0.0f, 1.0f);
mt19937 rng;
variate_generator<mt19937, normal_distribution<float> > var_nor(rng, _dist);
Matrix _mOriginalImage;

int main(int argc, char* argv[]){
	initTracker(".tif");
	runTracker();
	return 0;
}

void initTracker(char* ext){
	char* folder = "C:/Users/barry05/Desktop/Tracking Test Sequences/TiffSim";
	printf("Start Tracker...\n\nFolder: %s\n", folder);
	
	//Load file list
	vector<path> v = Utils::getFiles(folder);

	//Count files with specified extension
	int frames = Utils::countFiles(v, ext);
	vector<path>::iterator v_iter;
	
	//Get dimensions of first image
	int dims[2];
	Utils::getDims(v, ext, dims);

	//Construct image volume
	_mOriginalImage.width = dims[0];
	_mOriginalImage.stride = dims[0];
	_mOriginalImage.height = dims[1];
	_mOriginalImage.depth = frames;
	_mOriginalImage.size = dims[0] * dims[1] * frames;
	_mOriginalImage.elements = (float*)malloc(sizeof(float) * _mOriginalImage.size);

	//Load images into volume
	printf("\nLoading Images ... %d", 0);
	frames = 0;
	Mat frame;
	for(v_iter = v.begin(); v_iter != v.end(); v_iter++){
		string ext_s = ((*v_iter).extension()).string();
		if((strcmp(ext_s.c_str(), ext) == 0)) {
			printf("\rLoading Images ... %d", frames);
			frame = imread((*v_iter).string(), -1);
			Utils::copyToMatrix(frame, _mOriginalImage, frames);
			frames++;
		}
	}
	setup();
	return;
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
    _mSigmaPSFxy = (0.21f * _mWavelengthInNm / _mNA);
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
		int thisoffset = i * DIM_OF_STATE;
        float firstState[] = {x, y, 0.0f, 0.0f, 0.0f, 0.0f, mag};
		updateStateVector(firstState, i);
	}
	initParticleFilter(_mOriginalImage, _mInitRWIterations, _mFrameOfInitialization);
}

void initParticleFilter(Matrix aInitStack, int aInitParticleFilterIterations, int aFrameOfInit) {
        // - set up state vector
        // - create particles
        // - filter the initialized values
	_mParticles = (float*)malloc(sizeof(float) * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1));
    createParticles(_mStateVectors, _mParticles);
    filterTheInitialization(aInitStack, aInitParticleFilterIterations, aFrameOfInit);
	return;
}

void createParticles(float* aStateVectors, float* aParticles){
    for (int i=0; i < _currentLength; i++) {
		float* vState = &aStateVectors[i * DIM_OF_STATE];
        for (int vIndex = 0; vIndex < _mNbParticles; vIndex++) {
            float* vProposal = new float[DIM_OF_STATE + 1];
            for (int vI = 0; vI < DIM_OF_STATE; vI++) {
                vProposal[vI] = vState[vI];
            }
//				Init the weight as a last dimension
            vProposal[DIM_OF_STATE] = 1.0f; //not 0!			
            //add the new particle
            copyParticle(aParticles, vProposal, vIndex);
        }
    }
	return;
}

void filterTheInitialization(Matrix aImageStack, int aInitPFIterations, int aFrameOfInit){
	float vSigmaOfRWSave[DIM_OF_STATE];
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        vSigmaOfRWSave[vI] = _mSigmaOfRandomWalk[vI];
    }
    for (int vR = 0; vR < aInitPFIterations; vR++) {
        scaleSigmaOfRW(1.0f / powf(3.0f, (float)vR));
        DrawParticlesWithRW(_mParticles);

        updateParticleWeights(aImageStack, aFrameOfInit);

        estimateStateVectors(_mStateVectors, _mParticles);

        resample(_mParticles);
    }

    //restore the sigma vector
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        _mSigmaOfRandomWalk[vI] = vSigmaOfRWSave[vI];
    }
	return;
}

void scaleSigmaOfRW(float vScaler) {
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        _mSigmaOfRandomWalk[vI] *= vScaler;
    }
}

void DrawParticlesWithRW(float* aParticles) {
	for(int i=0; i<_currentLength;i++){
        for (int j=0; j<_mNbParticles; j++){
			int offset = i * _mNbParticles * (DIM_OF_STATE + 1);
			randomWalkProposal(aParticles, offset);
        }
	}
    return;
}

void randomWalkProposal(float* aParticles, int offset) {
    for (int aI = 0; aI < DIM_OF_STATE; aI++) {
        aParticles[aI] += var_nor() * _mSigmaOfRandomWalk[aI];
    }
	return;
}

void updateParticleWeights(Matrix aObservationStack, int aFrameIndex){
	for(int i=0; i<_currentLength; i++){
		int offset = i * _mNbParticles * (DIM_OF_STATE + 1);
        float vSumOfWeights = 0.0f;
        //
        // Calculate the likelihoods for each particle and save the biggest one
        //
        float* vLogLikelihoods = (float*)malloc(sizeof(float) * _mNbParticles);
        float vMaxLogLikelihood = -FLT_MAX;
		bool* vBitmap = (bool*)malloc(sizeof(bool) * _mWidth * _mHeight);
        generateParticlesIntensityBitmap(_mParticles, offset, vBitmap);
        calculateLikelihoods(aObservationStack, vBitmap, aFrameIndex, vLogLikelihoods, _mParticles, offset);
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
			int index = offset + vI * DIM_OF_STATE + DIM_OF_STATE;
            _mParticles[index] = _mParticles[index] * exp(vLogLikelihoods[vI]);
            vSumOfWeights += _mParticles[index];
            vI++;
        }
        //
        // Iterate again and normalize the weights
        //
        if (vSumOfWeights == 0.0f) { //can happen if the winning particle before had a weight of 0.0
			for (int vI = 0; vI < _mNbParticles; vI++) {
                _mParticles[offset + vI * DIM_OF_STATE + DIM_OF_STATE] = 1.0f / (float)_mNbParticles;
            }
        } else {
			for (int vI = 0; vI < _mNbParticles; vI++) {
                _mParticles[offset + vI * DIM_OF_STATE + DIM_OF_STATE] /= vSumOfWeights;
            }
        }
		free(vLogLikelihoods);
		free(vBitmap);
    }
	return;
}

void estimateStateVectors(float* aStateVectors, float* aParticles){
	return;
}

void resample(float* aParticles){
	return;
}

void generateParticlesIntensityBitmap(float* setOfParticles, int offset, bool* vBitmap) {
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
		int zero = offset + i * (DIM_OF_STATE + 1);
        if (setOfParticles[zero] - vMaxDistancexy < 0) {
            vXStart = 0;
        } else {
			vXStart = (int)boost::math::round<float>(setOfParticles[zero] - vMaxDistancexy);
        }
        if (setOfParticles[zero] + vMaxDistancexy >= _mWidth) {
            vXEnd = _mWidth - 1;
        } else {
            vXEnd = (int)boost::math::round<float>(setOfParticles[zero] + vMaxDistancexy);
        }
        if (setOfParticles[zero + 1] - vMaxDistancexy < 0) {
            vYStart = 0;
        } else {
            vYStart = (int)boost::math::round<float>(setOfParticles[zero + 1] - vMaxDistancexy);
        }
        if (setOfParticles[zero + 1] + vMaxDistancexy >= _mHeight) {
            vYEnd = _mHeight - 1;
        } else {
            vYEnd = (int)boost::math::round<float>(setOfParticles[zero + 1] + vMaxDistancexy);
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

void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int offset){
	//calculate ideal image
	for(int vI = offset; vI < offset + _mNbParticles; vI++){
		offset += DIM_OF_STATE + 1;
		float* vIdealImage = (float*)malloc(sizeof(float) * _mWidth * _mHeight);
		generateIdealImage(mParticles, offset, vIdealImage);
		//calculate likelihood
		vLogLikelihoods[vI] = calculateLogLikelihood(aObservationStack, aFrameIndex, vIdealImage, vBitmap);
		free(vIdealImage);
	}
	return;
}

void generateIdealImage(float* particles, int offset, float* vIdealImage) {
    addBackgroundToImage(vIdealImage, _mBackground);
    addFeaturePointToImage(vIdealImage, particles[0], particles[1], particles[6]);
    return;
}

void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity) {
    float vVarianceXYinPx = _mSigmaPSFxy * _mSigmaPSFxy / (_spatialRes * _spatialRes);
    float vMaxDistancexy = (3.0f * _mSigmaPSFxy / _spatialRes);

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
    if (x + .5f + (vMaxDistancexy + .5f) >= _mWidth) {
        vXEnd = _mWidth - 1;
    } else {
        vXEnd = (int) (x + .5f) + (int) (vMaxDistancexy + .5f);
    }
    if (y + .5f + (vMaxDistancexy + .5f) >= _mHeight) {
        vYEnd = _mHeight - 1;
    } else {
        vYEnd = (int) (y + .5f) + (int) (vMaxDistancexy + .5f);
    }
    for (int vY = vYStart; vY <= vYEnd && vY < _mHeight; vY++) {
		int woffset = vY * _mWidth;
        for (int vX = vXStart; vX <_mWidth; vX++) {
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
					vLogLikelihood += -aGivenImage[woffset + vX] + aStackProcs.elements[foffset + woffset + vX] * log(aGivenImage[woffset + vX]);
                }
            }
        }
//		System.out.println("used time for loglik = " + (System.currentTimeMillis() - vTime1));
    //IJ.showStatus("likelihood finshed");
    return vLogLikelihood;
}

//Copy statevectors corresponding to frame index from source to dest
void copyStateVector(float* dest, float* source, int index){
	int boffset = index * MAX_DETECTIONS * DIM_OF_STATE;
	for(int i=0; i < _currentLength; i++){
		 int soffset = i * DIM_OF_STATE;
		 for(int j=0; j<DIM_OF_STATE; j++){
			 dest[boffset + soffset + j] = source[boffset + soffset + j];
		 }
	}
	return;
}

void copyParticle(float* dest, float* source, int index){
	int boffset = index * MAX_DETECTIONS * (DIM_OF_STATE + 1);
	for(int i=0; i < DIM_OF_STATE + 1; i++){
		 dest[i + boffset] = source[i];
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