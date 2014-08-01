
#include "stdafx.h"
#include <tracker_tools.h>
#include <tracker_utils.h>
#include <tracker_copy_utils.h>
#include <defs.h>
#include <utils.h>
#include <global_params.h>
#include <gauss_tools.h>

void memErr();

//int main(int argc, char* argv[]){
//	int _mNbParticles = 100;
//	int _mInitRWIterations = 1;
//	char* _ext = ".tif";
//	char folder_c1[INPUT_LENGTH];
//	char folder_c2[INPUT_LENGTH];
//	float _numAp, _lambda;
//	bool verbose = true, maxDetectionsReached = false;
//	printf("Probabilistic Particle Tracker v1.%000d\n\n", getCurrentRevisionNumber(_tagFile, INPUT_LENGTH));
//	getParams(&_spatialRes, &_numAp, &_lambda, &_sigmaEstNM, &_sigmaEstPix, &_scalefactor, &_maxThresh, _ext, folder_c1, folder_c2, _configFile, &verbose);
//	checkFileSep(folder_c1);
//	bool mono = (strncmp(folder_c2, EMPTY, INPUT_LENGTH) == 0);
//	if(!mono){
//		checkFileSep(folder_c2);
//	}
//    string outputDir(folder_c1);
//    outputDir.append("/Cuda_Tracker_Output");
//    if(!(exists(outputDir))){
//        if(!(create_directory(outputDir))){
//            return -1;
//        }
//    }
//    int dims_c1[2], dims_c2[2];
//    clock_t start = clock();
//    printf("Start Tracker...\n\nFolder: %s\n", folder_c1);
//	
//    //Load file list
//    vector<path> v1 = getFiles(folder_c1);
//	vector<path> v2;
//	if(!mono) {
//		v2 = getFiles(folder_c2);
//	}
//
//    //Count files with specified extension
//    int numFiles = countFiles(v1, TIF);
//
//	if(!mono && numFiles != countFiles(v2, TIF)){
//		printf("\n\nFile number mismatch! Aborting...");
//		return 0;
//	}
//	
//    //Get dimensions of first channel1 image
//    getDims(v1, TIF, dims_c1);
//
//	//Get dimensions of first channel2 image (if applicable) and ensure they are the same as channel1
//    if(!mono){
//		getDims(v2, TIF, dims_c2);
//		if(dims_c1[0] != dims_c2[0] || dims_c1[1] != dims_c2[1]){
//			printf("\n\nImage dimension mismatch! Aborting...");
//			return 0;
//		}
//	}
//
//    //Construct channel1 image volume
//	Matrix _mOriginalImage_c1;
//    _mOriginalImage_c1.width = dims_c1[0];
//    _mOriginalImage_c1.height = dims_c1[1];
//	_mOriginalImage_c1.stride = _mOriginalImage_c1.width;
//    _mOriginalImage_c1.depth = numFiles;
//    _mOriginalImage_c1.size = _mOriginalImage_c1.width * _mOriginalImage_c1.height * numFiles;
//    _mOriginalImage_c1.elements = (float*)malloc(sizeof(float) * _mOriginalImage_c1.size);
//
//	//Construct channel2 image volume, if applicable
//	Matrix _mOriginalImage_c2;
//	if(!mono){
//		_mOriginalImage_c2.width = dims_c2[0];
//		_mOriginalImage_c2.height = dims_c2[1];
//		_mOriginalImage_c2.stride = _mOriginalImage_c2.width;
//		_mOriginalImage_c2.depth = numFiles;
//		_mOriginalImage_c2.size = _mOriginalImage_c2.width * _mOriginalImage_c2.height * numFiles;
//		_mOriginalImage_c2.elements = (float*)malloc(sizeof(float) * _mOriginalImage_c2.size);
//	} else {
//		_mOriginalImage_c2.size = 0;
//		_mOriginalImage_c2.elements = NULL;
//	}
//
//	int frames = 0;
//	if(!(_mOriginalImage_c1.elements == NULL)){
//		frames = loadImages(_mOriginalImage_c1, TIF, v1, folder_c1, numFiles, true);
//		if(!mono){
//			loadImages(_mOriginalImage_c2, TIF, v2, folder_c2, numFiles, true);
//		}
//	} else {
//		memErr();
//		return 0;
//	}
//
//    //Initialise memory spaces
//    float* _mParticles = (float*)malloc(sizeof(float) * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1));
//    float* _mStateVectorsMemory = (float*)malloc(sizeof(float) * _mOriginalImage_c1.depth * MAX_DETECTIONS * DIM_OF_STATE);
//    //float* _mParticlesMemory = (float*)malloc(sizeof(float) * _mOriginalImage_c1.depth * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1));
//	float* _mParticlesMemory = NULL;
//    float* _mStateVectors = (float*)malloc(sizeof(float) * MAX_DETECTIONS * DIM_OF_STATE);
//	//float* _mMaxLogLikelihood = (float*)malloc(sizeof(float) * _mOriginalImage.depth);
//    int* _counts = (int*)malloc(sizeof(int) * _mOriginalImage_c1.depth);
//
//	/*if(!(_mParticlesMemory == NULL) && !(_mParticles == NULL) && !(_mStateVectorsMemory == NULL)
//		&& !(_mStateVectors == NULL) && !(_counts == NULL)){*/
//	if(!(_mParticles == NULL) && !(_mStateVectorsMemory == NULL) && !(_mStateVectors == NULL) && !(_counts == NULL)){
//		Matrix candidates_c1;
//
//		// Find local maxima and use to initialise state vectors
//		candidates_c1.width = FIT_SIZE * MAX_DETECTIONS;
//		candidates_c1.stride = candidates_c1.width;
//		candidates_c1.height = FIT_SIZE + DATA_ROWS;
//		candidates_c1.size = candidates_c1.width * candidates_c1.height;
//		candidates_c1.elements = (float*) malloc(sizeof (float) * candidates_c1.size);
//		bool warnings[] = {true, false};
//		int _currentLength = maxFinder(NULL, _mOriginalImage_c1, candidates_c1, _maxThresh, false, 0, 0, 0, FIT_RADIUS, warnings, true);
//		if(_currentLength >= MAX_DETECTIONS){
//			_currentLength = MAX_DETECTIONS;
//			maxDetectionsReached = true;
//		}
//		for (int i = 0; i < _currentLength; i++) {
//			float x = candidates_c1.elements[i];
//			float y = candidates_c1.elements[i + candidates_c1.stride];
//			float mag = candidates_c1.elements[i * FIT_SIZE + FIT_RADIUS + candidates_c1.stride * (HEADER + FIT_RADIUS)];
//			float firstState[] = {x, y, 0.0f, 0.0f, 0.0f, 0.0f, mag};
//			updateStateVector(firstState, i, _mStateVectors);
//		}
//		free(candidates_c1.elements);
//		createParticles(_mStateVectors, _mParticles, _currentLength, _mNbParticles, 0);
//		copyStateParticlesToMemory(_mParticlesMemory, _mParticles, 0, _mNbParticles, _currentLength, 0);
//		filterTheInitialization(_mOriginalImage_c1, _mInitRWIterations, _mParticles, _currentLength, _mNbParticles, _mStateVectors, 0);
//		copyStateVector(_mStateVectorsMemory, _mStateVectors, 0, _currentLength);
//		maxDetectionsReached = maxDetectionsReached && runParticleFilter(_mOriginalImage_c1, _mParticles, _mParticlesMemory, _mStateVectors, _mStateVectorsMemory, _counts, _currentLength, _mNbParticles, _mInitRWIterations);
//
//		printf("\n\n");
//		output(dims_c1, frames, outputDir, _mNbParticles, _counts, _mParticlesMemory, _mStateVectorsMemory, _scalefactor, _mOriginalImage_c2, verbose, _maxThresh);
//		if(maxDetectionsReached){
//			printf("\n\nInsufficient memory allocated to track all particles.");
//		}
//		printf("\n\nElapsed Time: %.3f s\n", ((float)(clock() - start))/1000.0f);
//		waitForKey();
//	} else {
//		memErr();
//	}
//    return 0;
//}

void memErr(){
	printf("\n\nFailed to allocate sufficient memory - aborting.");
	waitForKey();
}