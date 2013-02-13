
#include "stdafx.h"
#include <tracker_utils.h>
#include <tracker_tools.h>
#include <matrix_mat.h>
#include <defs.h>
#include <boost/random.hpp>

void output(int* dims, int frames, string outputDir, int _mNbParticles, int* _counts, float* _mParticlesMemory, float* _mStateVectorsMemory, int _scalefactor, bool verbose) {
    FILE *datafile;
    FILE **datafilepointer = &datafile;
	string dataOutputDir(outputDir);
	const char* datafilename = (dataOutputDir.append("/tracker_data.txt")).data();
	fopen_s(datafilepointer, datafilename, "w");
	printf("Building output ... %d%%", 0);
    Matrix output;
    output.width = dims[0] * _scalefactor;
    output.stride = output.width;
    output.height = dims[1] * _scalefactor;
    output.size = output.width * output.height;
    output.elements = (float*) malloc(sizeof (float) * output.size);

	fprintf(datafile, "FRAMES %d\n", frames);
	fprintf(datafile, "TRAJECTORIES %d\n", _counts[frames - 1]);
	fprintf(datafile, "WIDTH %d\n", dims[0]);
	fprintf(datafile, "HEIGHT %d\n", dims[1]);

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
			float x = _mStateVectorsMemory[stateVectorIndex + _X_];
			float y = _mStateVectorsMemory[stateVectorIndex + _Y_];
			float mag = _mStateVectorsMemory[stateVectorIndex + _MAG_];
            addFeaturePointToImage(output.elements, x * _scalefactor, y * _scalefactor, mag, output.width, output.height);
			if(x >= 0.0 && y >= 0.0 && x < dims[0] && y < dims[1]){
				fprintf(datafile, "%f %f %f ", x, y, mag);
			} else {
				fprintf(datafile, "-1.0 -1.0 -1.0 ");
			}
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
		for (int j = _counts[frameIndex]; j < _counts[frames - 1]; j++){
			fprintf(datafile, "-1.0 -1.0 -1.0 ");
		}
		fprintf(datafile, "\n");
        copyFromMatrix(cudasaveframe, output, 0, 1.0f);
        for(int j = 1; j < frameIndex; j++){
                int stateVectorLastFrameIndex = (j - 1) * MAX_DETECTIONS * DIM_OF_STATE;
				int stateVectorNextFrameIndex = j * MAX_DETECTIONS * DIM_OF_STATE;
                for(int k = 0; k < _counts[j]; k++){
                        int stateVectorLastIndex = stateVectorLastFrameIndex + k * DIM_OF_STATE;
						int stateVectorNextIndex = stateVectorNextFrameIndex + k * DIM_OF_STATE;
						float x1 = _mStateVectorsMemory[stateVectorLastIndex + _X_];
						float y1 = _mStateVectorsMemory[stateVectorLastIndex + _Y_];
						float x2 = _mStateVectorsMemory[stateVectorNextIndex + _X_];
						float y2 = _mStateVectorsMemory[stateVectorNextIndex + _Y_];
						if(x1 > 0.0f && x1 < output.width && x2 > 0.0f && x2 < output.width && y1 > 0.0f && y1 < output.width && y2 > 0.0f && y2 < output.width){
							Point pt1((int) boost::math::round<float>(x1 * _scalefactor), (int) boost::math::round<float>(y1 * _scalefactor));
							Point pt2((int) boost::math::round<float>(x2 * _scalefactor), (int) boost::math::round<float>(y2 * _scalefactor));
							line(cudasaveframe, pt1, pt2, Scalar(255, 255, 255));
						}
                }
        }
        cudasaveframe.convertTo(cudasaveframe, CV_16UC1);
        string savefilename(outputDir);
        savefilename.append("/");
        savefilename.append(boost::lexical_cast<string > (frameIndex));
        savefilename.append(".tif");
        imwrite(savefilename, cudasaveframe);
    }
	fclose(datafile);
    return;
}
