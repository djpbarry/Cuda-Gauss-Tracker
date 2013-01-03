
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
#include <gauss_finder.h>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>

using namespace cv;

void runDetector();
int findParticles(Mat image, Matrix B, int count, int z);
//float testEvaluate(float x0, float y0, float max, int x, int y, float sig2);
float testGetRSquared(int x0, float srs, Matrix M);
bool testDraw2DGaussian(Matrix image, float x0, float y0, float prec);
bool drawSquare(Matrix image, float x01, float y01);
bool testDrawDot(Matrix image, float x01, float y01, float prec);
//void testDoMultiFit(Matrix M, int x0, int N, float *xe, float *ye, float *mag, float *r);
//float testSumMultiResiduals(int x0, float *xe, float *ye, float *mag, Matrix M, float xinc, float yinc, float minc, int index, int N);
float evaluate(float x0, float y0, int x, int y, float sigma2);
//int testInitialiseFitting(Matrix image, int index, int N, float *xe, float *ye, float *mag);
//void testCentreOfMass(float *x, float *y, int index, Matrix image);
float getPrecision(Matrix candidates, int index);

float _2sig2, _sig2;
float _maxThresh = 50.0f;
char* _ext = ".tif";
bool _lowRangeWarning = true, _highRangeWarning = false;

int main(int argc, char* argv[]) {
    runDetector();
    return 0;
}

void runDetector() {
    _spatialRes = getInput("spatial resolution in nm", _spatialRes);
    _maxThresh = getInput("maximum intensity threshold", _maxThresh);
    _lambda = getInput("wavelength of emitted light in nm", _lambda);
    _scalefactor = round(getInput("scaling factor for output", (float) _scalefactor));
    getTextInput("file extension", _ext);
    printf("\nSpatial resolution = %.0f nm", _spatialRes);
    printf("\nMaximum intensity threshold = %.0f", _maxThresh);
    printf("\nWavelength of emitted light = %.0f nm", _lambda);
    printf("\nOutput will be scaled by a factor of %d", _scalefactor);
    printf("\nFiles of type %s will be analysed", _ext);

    // Sigma estimate for Gaussian fitting
    _sigmaEst = 0.305f * _lambda / (_numAp * _spatialRes);
    _sig2 = _sigmaEst * _sigmaEst;
    _2sig2 = 2.0f * _sig2;

    printf("\n\nStart Detector...\n");
    char* folder = "C:/Users/barry05/Desktop/Test Data Sets/CUDA Gauss Localiser Tests/Test6";
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
                count = findParticles(frame, candidates, count, frames - (loopIndex * frameDiv));
                frames++;
            }
        }
        int width = frame.cols;
        int height = frame.rows;
        int outcount = 0;
        printf("\n\n-------------------------\n\nGPU Gaussian Fitting");
        if (_lowRangeWarning) {
            printf("\n\nWarning: GPU fitting may be unreliable due to small range of pixel values.\n\n");
        } else if (_highRangeWarning) {
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
							testDraw2DGaussian(cudaoutput, localisedX, localisedY, prec);
							//testDrawDot(cudaoutput, inputX, inputY, prec);
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

int findParticles(Mat image, Matrix B, int count, int frame) {
    Size radius(2 * FIT_RADIUS + 1, 2 * FIT_RADIUS + 1);
    Mat temp = image.clone();
    GaussianBlur(temp, image, radius, _sigmaEst, _sigmaEst);
    Matrix A;
    A.width = image.cols;
    A.stride = A.width;
    A.height = image.rows;
    A.size = A.width * A.height;
    A.elements = (float*) malloc(sizeof (float) * A.size);
    copyToMatrix(temp, A, 0);
    int thisCount = maxFinder(A, B, _maxThresh, true, count, 0, frame);
    free(A.elements);
    return thisCount;
}

float testEvaluate(float x0, float y0, float max, int x, int y, float sig2) {
    return max * expf(-((x - x0)*(x - x0) + (y - y0)*(y - y0)) / (2.0f * sig2));
}

float testGetRSquared(int x0, float srs, Matrix M) {
    int y0 = FIT_RADIUS + 3;
    float sumZ = 0.0f;
    for (int y = y0 - FIT_RADIUS; y <= y0 + FIT_RADIUS; y++) {
        int offset = y * M.stride;
        for (int x = x0 - FIT_RADIUS; x <= x0 + FIT_RADIUS; x++) {
            sumZ += M.elements[x + offset];
        }
    }
    float mean = sumZ / FIT_SIZE;
    float sumMeanDiffSqr = 0.0f;
    for (int y = y0 - FIT_RADIUS; y <= y0 + FIT_RADIUS; y++) {
        int offset = y * M.stride;
        for (int x = x0 - FIT_RADIUS; x <= x0 + FIT_RADIUS; x++) {
            sumMeanDiffSqr += (M.elements[x + offset] - mean) * (M.elements[x + offset] - mean);
        }
    }
    return 1.0f - srs / sumMeanDiffSqr;
}

//Searches for local maxima in A, greater in magnitude than maxThresh, and copies the local neighbourhood
//surrounding the maximum into B. Returns the total number of detected maxima in A.

extern "C" int maxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z) {
    float min, max, sum;
    int i, j;
    int koffset = k * A.width * A.height;
    for (int y = FIT_RADIUS; y < A.height - FIT_RADIUS; y++) {
        for (int x = FIT_RADIUS; x < A.width - FIT_RADIUS; x++) {
            for (min = FLT_MAX, max = -FLT_MAX, j = y - FIT_RADIUS; j <= y + FIT_RADIUS; j++) {
                int offset = koffset + j * A.stride;
                for (i = x - FIT_RADIUS; i <= x + FIT_RADIUS; i++) {
                    float pix = A.elements[i + offset];
                    _highRangeWarning = _highRangeWarning || (pix > 127.0f);
                    _lowRangeWarning = _lowRangeWarning && (pix < 1.0f);
                    if (!((x == i) && (y == j))) {
                        if (pix > max) {
                            max = pix;
                        }
                        if (pix < min) {
                            min = pix;
                        }
                    }
                }
            }
            float diff;
            float thispix = A.elements[x + y * A.stride];
            if (varyBG) {
                diff = thispix - min;
            } else {
                diff = thispix;
            }
            if ((thispix >= max) && (diff > maxThresh)) {
                int bxoffset = FIT_SIZE * count;
                sum = 0.0f;
                for (int m = y - FIT_RADIUS; m <= y + FIT_RADIUS; m++) {
                    int aoffset = m * A.stride;
                    int boffset = (m - y + FIT_RADIUS + 3) * B.stride;
                    for (int n = x - FIT_RADIUS; n <= x + FIT_RADIUS; n++) {
                        int bx = n - x + FIT_RADIUS;
                        sum += A.elements[aoffset + n];
                        B.elements[boffset + bx + bxoffset] = A.elements[aoffset + n];
                    }
                }
                B.elements[count] = (float) x;
                B.elements[count + B.stride] = (float) y;
                B.elements[count + 2 * B.stride] = (float) z;
                count++;
            }
        }
    }
    return count;
}

/*
void testDoMultiFit(Matrix M, int x0, int N, float *xe, float *ye, float *mag, float *r){
        for(int i=0; i<ITERATIONS; i++){
                for(int j=0; j<=N; j++){
                        float r1 = testSumMultiResiduals(x0, xe, ye, mag, M, -XY_STEP_SIZE, 0.0f, 0.0f, j, N);
                        float r2 = testSumMultiResiduals(x0, xe, ye, mag, M, XY_STEP_SIZE, 0.0f, 0.0f, j, N);
                        float r3 = testSumMultiResiduals(x0, xe, ye, mag, M, 0.0f, -XY_STEP_SIZE, 0.0f, j, N);
                        float r4 = testSumMultiResiduals(x0, xe, ye, mag, M, 0.0f, XY_STEP_SIZE, 0.0f, j, N);
                        float r5 = testSumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, -MAG_STEP_SIZE, j, N);
                        float r6 = testSumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, MAG_STEP_SIZE, j, N);
                        xe[N * N_MAX + j] -= (r2 - r1) * XY_STEP_SIZE / STEP_TOL;
                        ye[N * N_MAX + j] -= (r4 - r3) * XY_STEP_SIZE / STEP_TOL;
                        mag[N * N_MAX + j] -= (r6 - r5) * MAG_STEP_SIZE / STEP_TOL;
                        if(mag[N * N_MAX + j] < 0.0f) mag[N * N_MAX + j] = 0.0f;
                }
        }
        r[N] = testGetRSquared(x0, testSumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, 0.0f, 0, N), M);
        return;
}
 */

/*
float testSumMultiResiduals(int x0, float *xe, float *ye, float *mag, Matrix M, float xinc, float yinc, float minc, int index, int N) {
        float residuals = 0.0f;
                int eoffset = N * N_MAX;
        for (int j = Z_ROW + 1; j <= Z_ROW + FIT_SIZE; j++) {
                        int offset = j * M.stride;
            for (int i = x0 - FIT_RADIUS; i <= x0 + FIT_RADIUS; i++) {
                                float res = 0.0f;
                                int k;
                                for(k=0; k < index; k++){
                                        res += testMultiEvaluate(xe[k + eoffset], ye[k + eoffset], mag[k + eoffset], i, j);
                                }
                                res += testMultiEvaluate(xe[k + eoffset] + xinc, ye[k + eoffset] + yinc, mag[k + eoffset] + minc, i, j);
                                for(k = index + 1; k <= N; k++){
                                        res += testMultiEvaluate(xe[k + eoffset], ye[k + eoffset], mag[k + eoffset], i, j);
                                }
                                float e = res-M.elements[offset + i];
                residuals += e * e;
            }
        }
        return residuals;
    }
 */

/*float evaluate(float x0, float y0, float mag, int x, int y, float sigma2) {
    return mag * exp(-((x - x0)*(x - x0) + (y - y0)*(y - y0)) / (sigma2));
}*/

float evaluate(float x0, float y0, int x, int y, float sigma2) {
    return (1.0f / (2.0f * boost::math::constants::pi<float>() * sigma2)) * exp(-((x - x0)*(x - x0) + (y - y0)*(y - y0)) / (sigma2));
}

bool testDraw2DGaussian(Matrix image, float x01, float y01, float prec) {
    int x, y;
    float x0 = x01 * _scalefactor;
    float y0 = y01 * _scalefactor;
    float prec2s = (prec * prec);
    int drawRadius = (int) boost::math::round<float>(prec * 3.0f);
    if (drawRadius > min(0.1f * image.width, 0.1f * image.height)) return false;
    for (x = (int) floor(x0 - drawRadius); x <= x0 + drawRadius; x++) {
        for (y = (int) floor(y0 - drawRadius); y <= y0 + drawRadius; y++) {
            /* The current pixel value is added so as not to "overwrite" other
            Gaussians in close proximity: */
            if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
                int index = x + y * image.stride;
                image.elements[index] = image.elements[index] + evaluate(x0, y0, x, y, prec2s);
            }
        }
    }
    return true;
}

bool testDrawDot(Matrix image, float x01, float y01, float prec) {
    float x0 = x01 * _scalefactor;
    float y0 = y01 * _scalefactor;
	int index = x0 + y0 * image.stride;
    image.elements[index] = image.elements[index] + 1.0f;
    return true;
}

bool drawSquare(Matrix image, float x01, float y01) {
    int x, y;
    float x0 = x01 * _scalefactor;
    float y0 = y01 * _scalefactor;
    int drawRadius = FIT_RADIUS + 2;
    for (x = (int) floor(x0 - drawRadius); x <= x0 + drawRadius; x++) {
        y = (int) floor(y0 - drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
        y = (int) floor(y0 + drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
    }

    for (y = (int) floor(y0 - drawRadius); y <= y0 + drawRadius; y++) {
        x = (int) floor(x0 - drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
        x = (int) floor(x0 + drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
    }
    return true;
}

/*
int testInitialiseFitting(Matrix image, int index, int N, float *xe, float *ye, float *mag){
        testCentreOfMass(&xe[0], &ye[0], index, image);
        mag[0] = image.elements[index + (FIT_RADIUS + 3) * image.stride];
        float *r = (float*)malloc(sizeof(float) * N);
        testDoMultiFit(image, index, 0, xe, ye, mag, r);
        for(int n=1; n<N; n++){
                int noffset = n * N_MAX;
                mag[n + noffset] = 0.0f;
                xe[n + noffset] = 0.0f;
                ye[n + noffset] = 0.0f;
                for(int j=0; j<FIT_SIZE; j++){
                        int ioffset = (j+3) * image.stride;
                        for(int i=0; i<FIT_SIZE; i++){
                                float residual = image.elements[index - FIT_RADIUS + i + ioffset];
                                for(int m=0; m<n; m++){
                                        xe[m + noffset] = xe[m + (n-1) * N_MAX];
                                        ye[m + noffset] = ye[m + (n-1) * N_MAX];
                                        mag[m + noffset] = mag[m + (n-1) * N_MAX];
                                        residual -= testMultiEvaluate(xe[m + noffset], ye[m + noffset], mag[m + noffset], index - FIT_RADIUS + i, j+3);
                                }
                                if(residual > mag[n + noffset]){
                                        mag[n + noffset] = residual;
                                        xe[n + noffset] = (float)(i + index - FIT_RADIUS);
                                        ye[n + noffset] = j + 3.0f;
                                }
                        }
                }
                testDoMultiFit(image, index, n, xe, ye, mag, r);
        }
        int best = -1;
        float max = 0.0f;
        for(int i=0; i<N; i++){
                if(r[i] > max){
                        max = r[i];
                        best = i;
                }
        }
        return best;
}
 */

/*
void testCentreOfMass(float *x, float *y, int index, Matrix image){
        float xsum = 0.0f;
        float ysum = 0.0f;
        float sum = 0.0f;
        for(int j = 3; j<2*FIT_RADIUS+3+1; j++){
                int offset = j * image.stride;
                for(int i=index-FIT_RADIUS; i<=index+FIT_RADIUS; i++){
                        xsum += i * image.elements[i + offset];
                        ysum += j * image.elements[i + offset];
                        sum += image.elements[i + offset];
                }
        }
 *x = xsum / sum;
 *y = ysum / sum;
}
 */

float getPrecision(Matrix candidates, int index) {
    int y0 = HEADER;
    int x0 = index * FIT_SIZE;
    int best = round(candidates.elements[index + candidates.stride * BEST_ROW]);
    float bg = 0.0f;
    float N = 0.0f;
    for (int k = 0; k <= best; k++) {
        bg += candidates.elements[index + candidates.stride * (BG_ROW + k)];
        float mag = candidates.elements[index + candidates.stride * (MAG_ROW + k)];
        N += mag * (2.0f * boost::math::constants::pi<float>() * _sig2);
    }
    if (N > 0.0f) {
        bg /= (best + 1);
        float s = 2.0f * _sigmaEst * _spatialRes;
        float a = s * s + (_spatialRes * _spatialRes) / 12.0f;
        float b = 8.0f * boost::math::constants::pi<float>() * pow(s, 4.0f) * bg * bg;
        return pow((a / N) + (b / (_spatialRes * _spatialRes * N * N)), 0.5f);
    } else {
        return -1.0f;
    }
}