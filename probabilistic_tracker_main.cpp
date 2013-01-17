
#include <tracker_tools.h>

/*int main(int argc, char* argv[]){
	int _mNbParticles = 500;
	int _mInitRWIterations = 1;
	char* _ext = ".tif";
	char folder[INPUT_LENGTH];
	float _numAp, _lambda;
	printf("Probabilistic Particle Tracker v1.%000d\n\n", getCurrentRevisionNumber(_tagFile, INPUT_LENGTH));
	getParams(&_spatialRes, &_numAp, &_lambda, &_sigmaEstNM, &_sigmaEstPix, &_scalefactor, _ext, folder, _configFile);
    string outputDir(folder);
    outputDir.append("/CudaOutput");
    if(!(exists(outputDir))){
            if(!(create_directory(outputDir))){
                    return -1;
            }
    }
    int dims[2];
    clock_t start = clock();
    printf("Start Tracker...\n\nFolder: %s\n", folder);
	
    //Load file list
    vector<path> v = getFiles(folder);

    //Count files with specified extension
    int numFiles = countFiles(v, TIF);
	
    //Get dimensions of first image
    getDims(v, TIF, dims);

    //Construct image volume
	Matrix _mOriginalImage;
    _mOriginalImage.width = dims[0];
    _mOriginalImage.height = dims[1];
	_mOriginalImage.stride = _mOriginalImage.width;
    _mOriginalImage.depth = numFiles;
    _mOriginalImage.size = _mOriginalImage.width * _mOriginalImage.height * numFiles;
    _mOriginalImage.elements = (float*)malloc(sizeof(float) * _mOriginalImage.size);

    int frames = loadImages(_mOriginalImage, TIF, v, folder, numFiles, true);

    //Initialise memory spaces
    float* _mParticles = (float*)malloc(sizeof(float) * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1));
    float* _mStateVectorsMemory = (float*)malloc(sizeof(float) * _mOriginalImage.depth * MAX_DETECTIONS * DIM_OF_STATE);
    float* _mParticlesMemory = (float*)malloc(sizeof(float) * _mOriginalImage.depth * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1));
    float* _mStateVectors = (float*)malloc(sizeof(float) * MAX_DETECTIONS * DIM_OF_STATE);
	float* _mMaxLogLikelihood = (float*)malloc(sizeof(float) * _mOriginalImage.depth);
    int* _counts = (int*)malloc(sizeof(int) * _mOriginalImage.depth);

    Matrix candidates;

    // Find local maxima and use to initialise state vectors
    candidates.width = FIT_SIZE * MAX_DETECTIONS;
    candidates.stride = candidates.width;
    candidates.height = FIT_SIZE + DATA_ROWS;
    candidates.size = candidates.width * candidates.height;
    candidates.elements = (float*) malloc(sizeof (float) * candidates.size);

	bool warnings[] = {true, false};
    int _currentLength = maxFinder(_mOriginalImage, candidates, 0.0f, false, 0, 0, 0, FIT_RADIUS, warnings);
    for (int i = 0; i < _currentLength; i++) {
        float x = candidates.elements[i];
        float y = candidates.elements[i + candidates.stride];
        float mag = candidates.elements[i * FIT_SIZE + FIT_RADIUS + candidates.stride * (HEADER + FIT_RADIUS)];
        float firstState[] = {x, y, 0.0f, 0.0f, 0.0f, 0.0f, mag};
        updateStateVector(firstState, i, _mStateVectors);
    }
    free(candidates.elements);
    createParticles(_mStateVectors, _mParticles, _currentLength, _mNbParticles);
    copyStateParticlesToMemory(_mParticlesMemory, _mParticles, 0, _mNbParticles, _currentLength);
    filterTheInitialization(_mOriginalImage, _mInitRWIterations, _mParticles, _currentLength, _mNbParticles, _mStateVectors);
    copyStateVector(_mStateVectorsMemory, _mStateVectors, 0, _currentLength);
    runParticleFilter(_mOriginalImage, _mParticles, _mParticlesMemory, _mStateVectors, _mStateVectorsMemory, _counts, _currentLength, _mNbParticles);

    printf("\n\n");
    output(dims, frames, outputDir, _mNbParticles, _counts, _mParticlesMemory, _scalefactor);
    printf("\n\nElapsed Time: %.3f s\n", ((float)(clock() - start))/1000.0f);
	waitForKey();
    return 0;
}
*/