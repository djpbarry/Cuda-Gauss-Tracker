
#include "stdafx.h"
#include <tracker_utils.h>
#include <tracker_tools.h>
#include <matrix_mat.h>
#include <defs.h>
#include <boost/random.hpp>

void output(int* dims, int frames, string outputDir, int _mNbParticles, int* _counts, float* _mParticlesMemory, float* _mStateVectorsMemory, int _scalefactor) {
    printf("Building output ... %d%%", 0);
    Matrix output;
    output.width = dims[0] * _scalefactor;
    output.stride = output.width;
    output.height = dims[1] * _scalefactor;
    output.size = output.width * output.height;
    output.elements = (float*) malloc(sizeof (float) * output.size);

    for (int frameIndex = 0; frameIndex < frames; frameIndex++) {
        printf("\rBuilding output ... %d%%", (frameIndex + 1) * 100 / frames);
        //int stateVectorFrameIndex = frameIndex * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1);
		int stateVectorFrameIndex = frameIndex * MAX_DETECTIONS * DIM_OF_STATE;
        for (int p = 0; p < output.width * output.height; p++) {
            output.elements[p] = 0.0f;
        }
        Mat cudasaveframe(dims[1] * _scalefactor, dims[0] * _scalefactor, CV_32F);
        for (int i = 0; i < _counts[frameIndex]; i++) {
            //int stateVectorIndex = stateVectorFrameIndex + i * _mNbParticles * (DIM_OF_STATE + 1);
			int stateVectorIndex = stateVectorFrameIndex + i * DIM_OF_STATE;
            addFeaturePointToImage(output.elements, _mStateVectorsMemory[stateVectorIndex + _X_] * _scalefactor,
                    _mStateVectorsMemory[stateVectorIndex + _Y_] * _scalefactor,
                    _mStateVectorsMemory[stateVectorIndex + _MAG_], output.width, output.height);
            /*
			[DEBUG]
			for (int j = 0; j < _mNbParticles; j++) {
                int particleIndex = stateVectorIndex + j * (DIM_OF_STATE + 1);
                int x = (int) boost::math::round<float>(_mParticlesMemory[particleIndex] * _scalefactor);
                int y = (int) boost::math::round<float>(_mParticlesMemory[particleIndex + 1] * _scalefactor);
                //addFeaturePointToImage(output.elements, x, y, _mParticles[particleIndex + DIM_OF_STATE], output.width, output.height);
                if (x > 0 && x < output.width && y > 0 && y < output.height) {
                    int index = x + y * output.stride;
                    output.elements[index] += 1.0f;
                }
            }
			[/DEBUG]
			*/
        }
        copyFromMatrix(cudasaveframe, output, 0, 1.0f);
        /*for(int j = 0; j < frameIndex; j++){
                int stateVectorFrameIndex = j * MAX_DETECTIONS * DIM_OF_STATE;
                for(int k = 0; k < _counts[j]; k++){
                        int stateVectorIndex = stateVectorFrameIndex + k * DIM_OF_STATE;
						int x1 = (int) boost::math::round<float>(_mStateVectorsMemory[stateVectorIndex + _X_] * _scalefactor);
						int y1 = (int) boost::math::round<float>(_mStateVectorsMemory[stateVectorIndex + _Y_] * _scalefactor);
						//line(Mat& img, Point pt1, Point pt2, const Scalar& color);
                        circle(cudasaveframe, centre, FIT_RADIUS, 255);
                }
        }*/
        cudasaveframe.convertTo(cudasaveframe, CV_16UC1);
        string savefilename(outputDir);
        savefilename.append("/");
        savefilename.append(boost::lexical_cast<string > (frameIndex));
        savefilename.append(".tif");
        imwrite(savefilename, cudasaveframe);
    }
    return;
}
