
#include "stdafx.h"
#include <time.h>
#include <matrix_mat.h>
#include <defs.h>
#include <utils.h>
#include <iterator>
#include <vector>
#include <math.h>
#include <global_params.h>
#include <cuda_gauss_fitter.h>
#include <drawing.h>
#include <gauss_tools.h>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>

using namespace cv;

void runDetector();
bool loadParams(float *params, int paramcount, char *filename, int maxline, char *inputFolder);

int main(int argc, char* argv[]) {
    runDetector();
    return 0;
}

void runDetector() {
	float _2sig2, _sig2;
	float _maxThresh = 50.0f;
	char* _ext = ".tif";
	float params[NUM_PARAMS];
	char folder[INPUT_LENGTH];
	char tempExt[INPUT_LENGTH];
	if(!loadParams(params, NUM_PARAMS, "c:/users/barry05/gausstrackerparams.txt", INPUT_LENGTH, folder)){
		_spatialRes = params[0];
		_numAp = params[1];
		_lambda = params[2];
		_sigmaEst = params[3] * _lambda / (_numAp * _spatialRes);
		_scalefactor = (int)round(params[4]);
	}
    _spatialRes = getInput("spatial resolution in nm", _spatialRes);
    _maxThresh = getInput("maximum intensity threshold", _maxThresh);
    _lambda = getInput("wavelength of emitted light in nm", _lambda);
    _scalefactor = round(getInput("scaling factor for output", (float) _scalefactor));
    getTextInput("file extension", tempExt);

	if (tempExt[0] != '.') {
		printf("\n%s doesn't look like a valid file extension, so I'm going to look for %s files\n", tempExt, _ext);
    } else {
		strcpy_s(_ext, INPUT_LENGTH * sizeof (char), tempExt);
	}

    printf("\nSpatial resolution = %.0f nm", _spatialRes);
    printf("\nMaximum intensity threshold = %.0f", _maxThresh);
    printf("\nWavelength of emitted light = %.0f nm", _lambda);
    printf("\nOutput will be scaled by a factor of %d", _scalefactor);
    printf("\nFiles of type %s will be analysed", _ext);

    // Sigma estimate for Gaussian fitting
    _sigmaEst = 0.305f * _lambda / (_numAp * _spatialRes);
    _sig2 = _sigmaEst * _sigmaEst;
    _2sig2 = 2.0f * _sig2;
	bool warnings[] = {true, false};

    printf("\n\nStart Detector...\n");
    //char* folder = "C:/Users/barry05/Desktop/Test Data Sets/CUDA Gauss Localiser Tests/Test6";
    printf("\nFolder: %s\n", folder);

    string outputDir(folder);
    outputDir.append("/CudaOutput_NMAX");
    outputDir.append(boost::lexical_cast<string > (N_MAX));
    outputDir.append("_maxThresh");
    outputDir.append(boost::lexical_cast<string > (_maxThresh));
    if (!(exists(outputDir))) {
        if (!(create_directory(outputDir))) {
            return;
        }
    }

    vector<path> v = getFiles(folder);
    int frames = countFiles(v, _ext);
    vector<path>::iterator v_iter;

    v_iter = v.begin();

    float reqMem = ((float) FIT_SIZE) * MAX_DETECTIONS * frames * (FIT_SIZE + DATA_ROWS) * sizeof (float);
    int numLoops = (int) (ceilf(reqMem / MAX_MEMORY));
    int frameDiv = (int) (ceilf(((float) frames) / numLoops));
    int totalFrames = frames;
    frames = 0;
    int outFrames = 0;

    // Storage for regions containing candidate particles
    Matrix candidates;
    candidates.width = FIT_SIZE * MAX_DETECTIONS * frameDiv;
    candidates.stride = candidates.width;
    candidates.height = FIT_SIZE + DATA_ROWS;
    candidates.size = candidates.width * candidates.height;
    candidates.elements = (float*) malloc(sizeof (float) * candidates.size);

    Mat frame;
    for (int loopIndex = 0; loopIndex < numLoops; loopIndex++) {
        printf("\n\n-------------------------\n\nLOOP %d OF %d\n\n-------------------------\n\n", loopIndex + 1, numLoops);

        printf("\nFinding Maxima ... %d", 0);
        int count = 0;
        // Read one image at a time and find candidate particles in each
        for (; frames < (loopIndex + 1) * frameDiv && v_iter != v.end(); v_iter++) {
            string ext_s = ((*v_iter).extension()).string();
            if ((strcmp(ext_s.c_str(), _ext) == 0)) {
                printf("\rFinding Maxima ... %d", frames);
                frame = imread((*v_iter).string(), -1);
                count = findParticles(frame, candidates, count, frames - (loopIndex * frameDiv), FIT_RADIUS, _sigmaEst, _maxThresh, warnings);
                frames++;
            }
        }
        int width = frame.cols;
        int height = frame.rows;
        int outcount = 0;
        printf("\n\n-------------------------\n\nGPU Gaussian Fitting");
        if (warnings[0]) {
            printf("\n\nWarning: GPU fitting may be unreliable due to small range of pixel values.\n\n");
        } else if (warnings[1]) {
            printf("\n\nWarning: GPU fitting may be unreliable due to high range of pixel values.\n\n");
        }
        if (count > 0) GaussFitter(candidates, count, _sigmaEst, _maxThresh);
        clock_t totaltime = 0;
        printf("\n-------------------------\n\nWriting Output ... %d", 0);
        for (int z = 0; z < frameDiv; z++) {
            Matrix cudaoutput;
            cudaoutput.width = width*_scalefactor;
            cudaoutput.stride = cudaoutput.width;
            cudaoutput.height = height*_scalefactor;
            cudaoutput.size = cudaoutput.width * cudaoutput.height;
            cudaoutput.elements = (float*) malloc(sizeof (float) * cudaoutput.size);
            for (int p = 0; p < cudaoutput.width * cudaoutput.height; p++) {
                cudaoutput.elements[p] = 0.0f;
            }
            Mat cudasaveframe(height*_scalefactor, width*_scalefactor, CV_32F);
            while (round(candidates.elements[outcount + candidates.stride * Z_ROW]) <= z && outcount < count) {
                int inputX = round(candidates.elements[outcount + candidates.stride * X_ROW]);
                int inputY = round(candidates.elements[outcount + candidates.stride * Y_ROW]);
                int bestN = round(candidates.elements[outcount + candidates.stride * BEST_ROW]);
                int candidatesX = outcount * FIT_SIZE + FIT_RADIUS;
                int candidatesY = FIT_RADIUS + HEADER;
                if (bestN >= 0) {
                    for (int i = 0; i <= bestN; i++) {
                        float mag = candidates.elements[outcount + candidates.stride * (MAG_ROW + i)];
						float bg = candidates.elements[outcount + candidates.stride * (BG_ROW + i)];
						if(mag > bg){
							float localisedX = candidates.elements[outcount + candidates.stride * (XE_ROW + i)] + inputX - candidatesX;
							float localisedY = candidates.elements[outcount + candidates.stride * (YE_ROW + i)] + inputY - candidatesY;
							float prec = _sigmaEst * 100.0f / (mag - bg);
							draw2DGaussian(cudaoutput, localisedX * _scalefactor, localisedY * _scalefactor, prec);
							//testDrawDot(cudaoutput, inputX * _scalefactor, inputY * _scalefactor, prec);
						}
                    }
                }
                outcount++;
            }
            copyFromMatrix(cudasaveframe, cudaoutput, 0, 65536.0f);
            cudasaveframe.convertTo(cudasaveframe, CV_16UC1);
            printf("\rWriting Output ... %d", outFrames);
            string savefilename(outputDir);
            savefilename.append("/");
            savefilename.append(boost::lexical_cast<string > (outFrames));
            outFrames++;
            savefilename.append(PNG);
            imwrite(savefilename, cudasaveframe);
            free(cudaoutput.elements);
            cudasaveframe.release();
        }
        frame.release();
    }
    //printf("\n\nReference Time: %.0f", totaltime * 1000.0f/CLOCKS_PER_SEC);
    printf("\n\nPress Any Key...");
    getchar();
    getchar();
    return;
}

/*
	Load default parameter values from a configuration file.
*/
bool loadParams(float *params, int paramcount, char *filename, int maxline, char *inputFolder) {
    FILE *fp;
    FILE **fpp = &fp;
    fopen_s(fpp, filename, "r");
	char *line = (char*) malloc(maxline * sizeof(char));
	char *dummyString = (char*) malloc(maxline * sizeof(char));
	char dummyChar[1];
	int pindex = 0;
	if(fgets(line, maxline, fp) != NULL){
		sscanf_s(line, "%s %c %s", dummyString, maxline, dummyChar, sizeof(char), inputFolder, maxline);
	} else {
		return false;
	}
	while(fgets(line, maxline, fp) != NULL){
		sscanf_s(line, "%s %c %f", dummyString, maxline, dummyChar, sizeof(char), &params[pindex], sizeof(float));
		pindex++;
	}
    fclose(fp);
	if(pindex < paramcount){
		return false;
	} else {
		return true;
	}
}