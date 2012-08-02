#include "stdafx.h"
#include <matrix.h>
#include <utils.h>
#include <vector>
#include <iterator>

void runTracker(char* ext);
void setup();
void initialise();

int _mHeight;
int _mWidth;
int _mNFrames;
int _mTrackTillFrameNb;
float* _mStateVectorsMemory;
float* _mMaxLogLikelihood;
int _mFrameOfInitialization = 0;
float _mSigmaPSFxy;
float _mWavelengthInNm;
float _mn;
float _mNA;
float _mSigmaOfRandomWalk[] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
float _mBackground = 1.0f;
float _mSigmaOfDynamics[] = {20.0f, 20.0f, 20.0f, 1.0f};
bool _mDoPrecisionCorrection = true;
int _mDimOfState = 7;
Matrix _mOriginalImage;

int main(int argc, char* argv[]){
	runTracker(".tif");
	return 0;
}

void runTracker(char* ext){
	char* folder = "C:/Users/barry05/Desktop/CUDA Gauss Localiser Tests/Test6";
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
	initialise();
	return;
}

void setup() {
    _mHeight = _mOriginalImage.height;
    _mWidth = _mOriginalImage.width;
	_mNFrames = _mOriginalImage.depth;
    _mTrackTillFrameNb = _mNFrames;
    _mStateVectorsMemory = (float*)malloc(sizeof(float) * _mNFrames * MAX_DETECTIONS * _mDimOfState);
    _mMaxLogLikelihood = (float*)malloc(sizeof(float) * _mNFrames);
    _mSigmaPSFxy = (0.21f * _mWavelengthInNm / _mNA);
    return;
}

void initialise(){
	return;
}
