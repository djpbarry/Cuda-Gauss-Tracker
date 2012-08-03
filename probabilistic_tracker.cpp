#include "stdafx.h"
#include <matrix.h>
#include <utils.h>
#include <vector>
#include <iterator>

void initTracker(char* ext);
void setup();
void runTracker();
void initParticleFilter(Matrix aInitStack, int aInitParticleFilterIterations, int aFrameOfInit);
void createParticles(float* aStateVectors, float* aParticles);
void filterTheInitialization(Matrix aImageStack, int aInitPFIterations, int aFrameOfInit);
void scaleSigmaOfRW(float vScaler);
void DrawParticlesWithRW(float* aParticles);
void updateParticleWeights(Matrix aObservationStack, int aFrameIndex);
void estimateStateVectors(float* aStateVectors, float* aParticles);
void resample(float* aParticles);
void copyStateVector(float* dest, float* source, int index);
void copyParticle(float* dest, float* source, int index);
void updateStateVector(float* vector, int index);
extern int maxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z);

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
float _mWavelengthInNm;
float _mn;
float _mNA;
float _mSigmaOfRandomWalk[] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
float _mBackground = 1.0f;
float _mSigmaOfDynamics[] = {20.0f, 20.0f, 20.0f, 1.0f};
bool _mDoPrecisionCorrection = true;
int _currentLength;
int _mNbParticles = 10;
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
        scaleSigmaOfRW(1.0f / powf(3, vR));
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
    return;
}

void updateParticleWeights(Matrix aObservationStack, int aFrameIndex){
	return;
}

void estimateStateVectors(float* aStateVectors, float* aParticles){
	return;
}

void resample(float* aParticles){
	return;
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