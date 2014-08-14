
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
#include <jni.h>
#include <ParticleTracking_Timelapse_Analysis.h>

using namespace cv;

bool runDetector(const char* folder, const char* ext, float spatialRes, float sigmaEst, float maxThresh, float fitTol, int start, int end);

JNIEXPORT jboolean JNICALL Java_ParticleTracking_Timelapse_1Analysis_cudaGaussFitter
(JNIEnv *env, jobject obj, jstring folder, jstring ext, jfloat spatialRes, jfloat sigmaEst, jfloat maxThresh, jfloat fitTol, jint start, jint end) {
    const char *cfolder = env->GetStringUTFChars(folder, NULL);
    if (NULL == cfolder) {
        printf("Invalid folder specified.");
        return (jboolean) false;
    }

    const char *cext = env->GetStringUTFChars(ext, NULL);
    if (NULL == cext) {
        printf("Invalid file extension specified.");
        return (jboolean) false;
    }

    jboolean result = runDetector(cfolder, cext, spatialRes, sigmaEst, maxThresh, fitTol, start, end);

    env->ReleaseStringUTFChars(folder, cfolder);
    env->ReleaseStringUTFChars(ext, cext);

    return result;
}

//int main(int argc, char* argv[]) {
//    runDetector("C:/Users/barry05/Desktop/SuperRes Actin Tails/2014.07.08_Dualview/Lifeact/Capture_1/C0",
//		".tif", 133.333f, 1.06f, 0.99f, 0.95f, 0, 499);
//    return 0;
//}

bool runDetector(const char* folder, const char* ext, float spatialRes, float sigmaEst, float percentThresh, float fitTol, int start, int end) {
    //float _2sig2, _sig2, _numAp, _lambda, percentThresh, fitTol =0.95f;
    //char* _ext = ".tif";
    bool verbose = true;
    bool drawDots = false;
    //char folder_c1[INPUT_LENGTH], folder_c2[INPUT_LENGTH];
    //getParams(&_spatialRes, &_numAp, &_lambda, &_sigmaEstNM, &_sigmaEstPix, &_scalefactor, &percentThresh, _ext, folder_c1, folder_c2, _configFile, &verbose);

    // Sigma estimate for Gaussian fitting
    _sigmaEstNM = sigmaEst;
    //_sig2 = _sigmaEstNM * _sigmaEstNM;
    //_2sig2 = 2.0f * _sig2;
    bool warnings[] = {true, false};

    string dataDir(folder);
    dataDir.append("/cudadata.txt");
    FILE *data;
    FILE **pdata = &data;
    fopen_s(pdata, dataDir.data(), "w");

    string logDir(folder);
    logDir.append("/cudalog.txt");
    FILE *log;
    FILE **plog = &log;
    fopen_s(plog, logDir.data(), "w");

    fprintf(log, "Start Detector...\n\n");
    //char* folder = "C:/Users/barry05/Desktop/Test Data Sets/CUDA Gauss Localiser Tests/Test6";
    fprintf(log, "Folder: %s\n\n", folder);

    //string outputDir(folder);
    //outputDir.append("/CudaOutput_NMAX");
    //outputDir.append(boost::lexical_cast<string > (N_MAX));
    //outputDir.append("percentThresh");
    //outputDir.append(boost::lexical_cast<string > (percentThresh));
    //if (!(exists(outputDir))) {
    //    if (!(create_directory(outputDir))) {
    //        return false;
    //    }
    //}

    vector<path> v = getFiles(folder);
    int frames = countFiles(v, ext);
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

    if (candidates.elements == NULL) {
        fprintf(log, "Failed to allocate memory - aborting.\n\n");
        return false;
    } else {
        fprintf(log, "Memory allocated - proceeding...\n\n", folder);
    }

    Mat frame;
    //fprintf(data, "%s %s %f %f %f %f\n\n", folder, ext, spatialRes, sigmaEst, percentThresh, fitTol);
    for (int loopIndex = 0; loopIndex < numLoops; loopIndex++) {
        fprintf(log, "-------------------------\n\nLOOP %d OF %d\n\n-------------------------\n\n", loopIndex + 1, numLoops);

        fprintf(log, "Finding Maxima ...\n\n");
        int count = 0;
        // Read one image at a time and find candidate particles in each
        for (; frames < (loopIndex + 1) * frameDiv && v_iter != v.end(); v_iter++) {
            string ext_s = ((*v_iter).extension()).string();
            if ((strcmp(ext_s.c_str(), ext) == 0)) {
                if (frames >= start && frames <= end) {
                    printf("\rFinding Maxima ... %d", frames);
                    frame = imread((*v_iter).string(), -1);
                    //_maxThresh = getPercentileThresh(&frame, percentThresh);
                    count = findParticles(frame, candidates, count, frames - (loopIndex * frameDiv), FIT_RADIUS, _sigmaEstNM, percentThresh, warnings, true);
                    if (count < 0) {
                        fprintf(log, "Too many maxima! Aborting.\n\n");
                        return false;
                    }
                }
                frames++;
            }
        }
        int width = frame.cols;
        int height = frame.rows;
        int outcount = 0;
        fprintf(log, "Gaussian Fitting...\n\n");
        if (warnings[0]) {
            fprintf(log, "Warning: GPU fitting may be unreliable due to small range of pixel values.\n\n");
        } else if (warnings[1]) {
            fprintf(log, "Warning: GPU fitting may be unreliable due to high range of pixel values.\n\n");
        }
        if (count > 0) GaussFitter(candidates, count, _sigmaEstNM);
        clock_t totaltime = 0;
        //printf("\n-------------------------\n\nWriting Output ... %d", 0);
        for (int z = 0; z < frameDiv; z++) {
            //Matrix cudaoutput;
            //cudaoutput.width = width*_scalefactor;
            //cudaoutput.stride = cudaoutput.width;
            //cudaoutput.height = height*_scalefactor;
            //cudaoutput.size = cudaoutput.width * cudaoutput.height;
            //cudaoutput.elements = (float*) malloc(sizeof (float) * cudaoutput.size);
            //for (int p = 0; p < cudaoutput.width * cudaoutput.height; p++) {
            //    cudaoutput.elements[p] = 0.0f;
            //}
            //Mat cudasaveframe(height*_scalefactor, width*_scalefactor, CV_32F);
            while (round(candidates.elements[outcount + candidates.stride * Z_ROW]) <= z && outcount < count) {
                int inputX = round(candidates.elements[outcount + candidates.stride * X_ROW]);
                int inputY = round(candidates.elements[outcount + candidates.stride * Y_ROW]);
                int bestN = round(candidates.elements[outcount + candidates.stride * BEST_ROW]);
                int candidatesX = outcount * FIT_SIZE + FIT_RADIUS;
                int candidatesY = FIT_RADIUS + HEADER;
                if (bestN >= 0 && candidates.elements[outcount + candidates.stride * R_ROW] > fitTol) {
                    for (int i = 0; i <= bestN; i++) {
                        float mag = candidates.elements[outcount + candidates.stride * (MAG_ROW + i)];
                        float bg = candidates.elements[outcount + candidates.stride * (BG_ROW + i)];
                        //if(mag > bg){
                        float localisedX = candidates.elements[outcount + candidates.stride * (XE_ROW + i)] + inputX - candidatesX;
                        float localisedY = candidates.elements[outcount + candidates.stride * (YE_ROW + i)] + inputY - candidatesY;
                        float r2 = candidates.elements[outcount + candidates.stride * (R_ROW)];
                        //float prec = _sigmaEstNM * 100.0f / (mag - bg);
                        //float prec = 1.0f;
                        //float prec = _sigmaEstNM / _spatialRes;
                        //if (drawDots) {
                        //    drawDot(cudaoutput, localisedX * _scalefactor, localisedY * _scalefactor);
                        //} else {
                        //    draw2DGaussian(cudaoutput, localisedX * _scalefactor, localisedY * _scalefactor, _sigmaEstNM);
                        //
                        fprintf(data, "%d %f %f %f %f\n", outFrames, localisedX * spatialRes / 1000.0, localisedY * spatialRes / 1000.0, mag, r2);
                        //testDrawDot(cudaoutput, inputX * _scalefactor, inputY * _scalefactor, prec);
                        //}
                    }
                }
                outcount++;
            }
            //copyFromMatrix(cudasaveframe, cudaoutput, 0, 65536.0f);
            //cudasaveframe.convertTo(cudasaveframe, CV_16UC1);
            //printf("\rWriting Output ... %d", outFrames);
            //string savefilename(outputDir);
            //savefilename.append("/");
            //savefilename.append(boost::lexical_cast<string > (outFrames));
            outFrames++;
            //savefilename.append(PNG);
            //imwrite(savefilename, cudasaveframe);
            //free(cudaoutput.elements);
            //cudasaveframe.release();
        }
        frame.release();
        fprintf(log, "LOOP %d done.\n\n", loopIndex + 1);
    }
    fclose(data);
    fclose(log);
    candidates.elements = NULL;
    //printf("\n\nReference Time: %.0f", totaltime * 1000.0f/CLOCKS_PER_SEC);
    //printf("\n\nPress Any Key...");
    //waitForKey();
    return true;
}
