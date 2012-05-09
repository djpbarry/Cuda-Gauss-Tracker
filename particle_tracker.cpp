// ParticleTracker.cpp : Defines the entry point for the console application.
//
// includes, system
//#include"particlearray.cpp"
#include "stdafx.h"
//#include "gaussianfitter.cpp"
//#include <math.h>
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <matrix.h>

using namespace cv;

//void analyse(Mat, Mat);

int findParticles(Mat image, Matrix B, int count, int z);

//int maxFinder(Mat, int, int, float, float, bool, int*, int*);

extern "C" float GaussFitter(Matrix A, int maxcount, float sigEst, float maxThresh);

void copyToMatrix(Mat, Matrix);

void copyFromMatrix(Mat, Matrix);

char* toString(int);

int round(float);

int testMaxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int z);

void testMaxFinderKernel(const Matrix, Matrix, const float, bool, float, int, int); 

float testSumResiduals(int x0, int y0, float xe, float ye, float max, Matrix M, float sig2);

float testEvaluate(float x0, float y0, float max, int x, int y, float sig2);

float testGetRSquared(int x0, float srs, Matrix M);

void testDoMEFit(Matrix M, int x0, int y0, float max, float sig2, float *results);

bool testDraw2DGaussian(Matrix image, float mag, float x0, float y0);

void testDoMultiFit(Matrix M, int x0, int N, float *xe, float *ye, float *mag, float *r);

float testSumMultiResiduals(int x0, float *xe, float *ye, float *mag, Matrix M, float xinc, float yinc, float minc, int index, int N);

float testMultiEvaluate(float x0, float y0, float mag, int x, int y);

int initialiseFitting(Matrix image, int index, int N, float *xe, float *ye, float *mag);

void testCentreOfMass(float *x, float *y, int index, Matrix image);

int factorial(int i);

float _spatialRes = 132.0f;
float _sigmaEst, _2sig2;
float _maxThresh = 5.0f;
float _numAp = 1.4f;
float _lambda = 650.0f;
int _numFrames = 150;
int _scalefactor = 1;

int main(int argc, char* argv[])
{
	// Sigma estimate for Gaussian fitting
	_sigmaEst = 0.305f * _lambda / (_numAp * _spatialRes);
	_2sig2 = 2.0f * _sigmaEst * _sigmaEst;

	printf("Start...\n");
	String folder = "C:\\Users\\barry05\\Desktop\\Tracking Test Sequences\\OpenCV Tests\\Test6\\";
	printf("\nFolder: %s\n", folder);
	
	// Names of first files is folder
	String filename = "000.tif", savefilename = "000.tif";
	Mat frame = imread(folder + filename, -1);
	int width = frame.cols;
	int height = frame.rows;
	int frames = 0;
	int count = 0;

	// Storage for regions containing candidate particles
	Matrix candidates;

	// Width of each region set to one quarter warp (assuming FIT_RADIUS = 3)
	candidates.width = FIT_SIZE * MAX_DETECTIONS * _numFrames;
	candidates.stride = candidates.width;
	candidates.height = FIT_SIZE + DATA_ROWS;
	candidates.size = candidates.width * candidates.height;
	candidates.elements = (float*)malloc(sizeof(float) * candidates.size);

	// Read one image at a time and find candidate particles in each
	while(!frame.empty()){
		//printf("\n%d: %d", frames, count);
		count = findParticles(frame, candidates, count, frames);
		frames++;
		filename = strcat(toString(frames),TIF);
		float noughts = log10((float)frames);
		while(noughts < 2){
			filename = '0' + filename;
			noughts += 1.0f;
		}
		frame = imread(folder + filename, -1);
	}
	int outcount = 0;
	savefilename = "000.tif";
	GaussFitter(candidates, count, _sigmaEst, _maxThresh);
	clock_t totaltime = 0;
	for(int z=0; z<frames; z++){
		if(z>0){
			savefilename = strcat(toString(z),TIF);
			float noughts = log10((float)z);
			while(noughts < 2){
				savefilename = '0' + savefilename;
				noughts += 1.0f;
			}
		}
		Matrix cudaoutput;
		cudaoutput.width = width;
		cudaoutput.stride=cudaoutput.width;
		cudaoutput.height = height;
		cudaoutput.size = cudaoutput.width * cudaoutput.height;
		cudaoutput.elements = (float*)malloc(sizeof(float) * cudaoutput.size);
		for(int p=0; p<width*height; p++){
			cudaoutput.elements[p] = 0.0f;
		}
		//Mat cudaoutframe(height,width,CV_32F);
		Mat cudasaveframe(height*_scalefactor,width*_scalefactor,CV_32F);
		Matrix testoutput;
		testoutput.width = width*_scalefactor;
		testoutput.stride=testoutput.width;
		testoutput.height = height*_scalefactor;
		testoutput.size = cudaoutput.width * cudaoutput.height;
		testoutput.elements = (float*)malloc(sizeof(float) * testoutput.size);
		for(int p=0; p<width*height*_scalefactor*_scalefactor; p++){
			testoutput.elements[p] = 0.0f;
		}
		//Mat testoutframe(height,width,CV_32F);
		Mat testsaveframe(height*_scalefactor,width*_scalefactor,CV_32F);
		while(round(candidates.elements[outcount + candidates.stride * Z_ROW]) <= z && outcount<count){
			int x = round(candidates.elements[outcount + candidates.stride * X_ROW]);
			int y = round(candidates.elements[outcount + candidates.stride * Y_ROW]);
			int best = round(candidates.elements[outcount + candidates.stride * BEST_ROW]);
			int xRegionCentre = outcount*FIT_SIZE+FIT_RADIUS;
			int yRegionCentre = FIT_RADIUS+HEADER;
			if(best>=0){
				//printf("\nz: %d, outcount: %d", z, outcount);
				for(int i=0; i<=best; i++){
					//printf("\nxe: %f ye: %f mag: %f\n",x + candidates.elements[outcount + candidates.stride * (XE_ROW + i)] - xRegionCentre,y + candidates.elements[outcount + candidates.stride * (YE_ROW + i)] - yRegionCentre,candidates.elements[outcount + candidates.stride * (MAG_ROW + i)]);
					testDraw2DGaussian(cudaoutput, candidates.elements[outcount + candidates.stride * (MAG_ROW + i)],
						x + candidates.elements[outcount + candidates.stride * (XE_ROW + i)] - xRegionCentre,
						y + candidates.elements[outcount + candidates.stride * (YE_ROW + i)] - yRegionCentre);
					/*printf("x: %d, y: %d, z: %d, xinc: %f, yinc: %f, mag: %f\n", x, y, z,
						candidates.elements[outcount + candidates.stride * (XE_ROW + i)] - xRegionCentre,
						candidates.elements[outcount + candidates.stride * (YE_ROW + i)] - yRegionCentre,
						candidates.elements[outcount + candidates.stride * (MAG_ROW + i)]);*/
				}
			}
			float *xe = (float*)malloc(sizeof(float) * N_MAX * N_MAX);
			float *ye = (float*)malloc(sizeof(float) * N_MAX * N_MAX);
			float *mag = (float*)malloc(sizeof(float) * N_MAX * N_MAX);
			clock_t start = clock();
			best = initialiseFitting(candidates, xRegionCentre, N_MAX, xe, ye, mag);
			totaltime += clock() - start;
			if(best>=0){
				//printf("\nz: %d, outcount: %d", z, outcount);
				for(int i=0; i<=best; i++){
					//printf("\nxe: %f ye: %f mag: %f\n",x + xe[best * N_MAX + i] - xRegionCentre,y + ye[best * N_MAX + i] - yRegionCentre,mag[best * N_MAX + i]);
					testDraw2DGaussian(testoutput, mag[best * N_MAX + i], x + xe[best * N_MAX + i] - xRegionCentre, y + ye[best * N_MAX + i] - yRegionCentre);
				}
			}
			outcount++;
		}
		copyFromMatrix(cudasaveframe, cudaoutput);
		//cudaoutframe.convertTo(cudasaveframe,CV_8U);
		imwrite(folder + "CudaOutput\\" + savefilename, cudasaveframe);
		copyFromMatrix(testsaveframe, testoutput);
		//testoutframe.convertTo(testsaveframe,CV_8U);
		imwrite(folder + "TestOutput\\" + savefilename, testsaveframe);
	}
	char c;
	printf("\n\nReference Time: %.0f", totaltime * 1000.0f/CLOCKS_PER_SEC);
	printf("\n\nPress Any Key...");
	scanf_s("%c",&c);
	return 0;
}

int findParticles(Mat image, Matrix B, int count, int frame) {
	Size radius(2*FIT_RADIUS+1, 2*FIT_RADIUS+1);
	Mat temp = image.clone();
	GaussianBlur(temp, image, radius, _sigmaEst, _sigmaEst);
	Matrix A;
	A.width = image.cols;
	A.stride = A.width;
	A.height = image.rows;
	A.size = A.width * A.height;
	A.elements = (float*)malloc(sizeof(float) * A.size);
	copyToMatrix(temp, A);
	return testMaxFinder(A, B, _maxThresh, true, count, frame);
}

void copyToMatrix(Mat M, Matrix A){
	for(int y=0; y< M.rows; y++){
		int Moffset = y * M.step1();
		int Aoffset = y * A.stride;
		for(int x=0; x < M.cols; x++){
			float a = ((float*)M.data)[Moffset + x];
			A.elements[Aoffset + x] = ((float*)M.data)[Moffset + x];
		}
	}
	return;
}

void copyFromMatrix(Mat M, Matrix A){
	//printf("\nOutput: \n");
	for(int y=0; y< M.rows; y++){
		int Moffset = y * M.step1();
		int Aoffset = y * A.stride;
		for(int x=0; x < M.cols; x++){
			float a = A.elements[Aoffset + x];
			((float*)M.data)[Moffset + x] = A.elements[Aoffset + x];
			//printf(" | %.1f | ", a);
		}
		//printf("\n");
	}
	return;
}

char* toString(int i){
	int l = (int)floor(log10((float)i));
	l++;
	char *out = (char*) malloc (sizeof(char) * (l + 1));
	for(int j=l-1; j>=0; j--){
		out[j] = (char)(i % 10 + 48);
		i /= 10;
	}
	out[l] = '\0';
	return out;
}

float testSumResiduals(int x0, int y0, float xe, float ye, float max, Matrix M, float sig2) {
        float residuals = 0.0f;
        for (int j = y0 - FIT_RADIUS; j <= y0 + FIT_RADIUS; j++) {
			int offset = j * M.stride;
            for (int i = x0 - FIT_RADIUS; i <= x0 + FIT_RADIUS; i++) {
				float e = testEvaluate(xe, ye, max, i, j, sig2) - M.elements[offset + i];
                residuals += e * e;
            }
        }
        return residuals;
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

// Matrix multiplication - Host code 
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
int testMaxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int z) 
{ 
	float min, max;
	int i, j;
	int blocksize = FIT_SIZE;
	for(int y=FIT_RADIUS; y<A.height - FIT_RADIUS; y++){
		for(int x=FIT_RADIUS; x<A.width - FIT_RADIUS; x++){
			for (min = 255.0, max = 0.0, i = x - FIT_RADIUS; i <= x + FIT_RADIUS; i++) {
				for (j = y - FIT_RADIUS; j <= y + FIT_RADIUS; j++){
					if ((A.elements[i + j * A.stride] > max) && !((x == i) && (y == j))) {
						max = A.elements[i + j * A.stride];
					}
					if ((A.elements[i + j * A.stride] < min) && !((x == i) && (y == j))) {
						min = A.elements[i + j * A.stride];
					}
				}
			}
			float diff;
			float thispix = A.elements[x + y * A.stride];
			if (varyBG) {
				diff = thispix- min;
			} else {
				diff = thispix;
			}
			if ((thispix >= max) && (diff > maxThresh)) {
				int bxoffset = blocksize * count;
				for(int m=y-FIT_RADIUS; m <= y+FIT_RADIUS; m++){
					int aoffset = m * A.stride;
					int boffset = (m - y + FIT_RADIUS+3) * B.stride;
					for(int n=x-FIT_RADIUS; n <= x+FIT_RADIUS; n++){
						int bx = n - x + FIT_RADIUS;
						B.elements[boffset + bx + bxoffset] = A.elements[aoffset + n] - min;
					}
				}
				B.elements[count] = (float)x;
				B.elements[count + B.stride] = (float)y;
				B.elements[count + 2 * B.stride] = (float)z;
				count++;
			}
		}
	}
	return count;
}

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
		/*printf("\n\nIteration: %d, xe[0]: %f, xe[1]: %f, xe[2]: %f, xe[3]: %f, ye[0]: %f, ye[1]: %f, ye[2]: %f, ye[3]: %f, mag[0]: %f, mag[1]: %f, mag[2]: %f, mag[3]: %f",
			i, xe[0], xe[1], xe[2], xe[3], ye[0], ye[1], ye[2], ye[3], mag[0], mag[1], mag[2], mag[3]);*/
	}
	r[N] = testGetRSquared(x0, testSumMultiResiduals(x0, xe, ye, mag, M, 0.0f, 0.0f, 0.0f, 0, N), M);
	return;
}

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

float testMultiEvaluate(float x0, float y0, float mag, int x, int y) {
		return mag * exp(-((x - x0)*(x - x0) + (y - y0)*(y - y0)) / (_2sig2));
    }

int round(float number)
{
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}

bool testDraw2DGaussian(Matrix image, float mag, float x01, float y01) {
        int x, y;
        float value;
		float x0 = x01 * _scalefactor;
		float y0 = y01 * _scalefactor;
		int drawRadius = FIT_RADIUS + 2;
        for (x = (int) floor(x0 - drawRadius); x <= x0 + drawRadius; x++) {
            for (y = (int) floor(y0 - drawRadius); y <= y0 + drawRadius; y++) {
                /* The current pixel value is added so as not to "overwrite" other
                Gaussians in close proximity: */
				if(x >= 0 && x < image.width && y >= 0 && y < image.height){
					//value = mag * exp(-(((x - x0) * (x - x0)) + ((y - y0) * (y - y0))) / (2.0f * _sig2));
					//value += image.elements[x + y * image.stride];
					image.elements[x + y * image.stride] = image.elements[x + y * image.stride] + testMultiEvaluate(x0, y0, mag, x, y);
				}
            }
        }
        return true;
    }

int initialiseFitting(Matrix image, int index, int N, float *xe, float *ye, float *mag){
	testCentreOfMass(&xe[0], &ye[0], index, image);
	mag[0] = image.elements[index + (FIT_RADIUS + 3) * image.stride];
	float *r = (float*)malloc(sizeof(float) * N);
	testDoMultiFit(image, index, 0, xe, ye, mag, r);
	for(int n=1; n<N; n++){
		int noffset = n * N_MAX;
		/*Matrix estimates;
		estimates.width = FIT_SIZE;
		estimates.height = FIT_SIZE;
		estimates.stride = estimates.width;
		estimates.size = estimates.width * estimates.height;
		estimates.elements = (float*)malloc(sizeof(float) * estimates.width * estimates.height);
		for(int j=0; j<estimates.height; j++){
			int roffset = j * estimates.stride;
			for(int i=0; i<estimates.height; i++){
				estimates.elements[i + roffset] = 0.0f;
			}
		}
		for(int m=0; m<n; m++){
			xe[m + noffset] = xe[m + (n-1) * N_MAX];
			ye[m + noffset] = ye[m + (n-1) * N_MAX];
			mag[m + noffset] = mag[m + (n-1) * N_MAX];
			//printf("\n\nm: %d, xe[m + noffset]: %f, ye[m + noffset]: %f, mag[m + noffset]: %f",m, xe[m + noffset], ye[m + noffset], mag[m + noffset]);
			testDraw2DGaussian(estimates, mag[m + noffset], xe[m + noffset]-index+FIT_RADIUS, ye[m + noffset]-3.0f);
		}*/
		mag[n + noffset] = 0.0f;
		xe[n + noffset] = 0.0f;
		ye[n + noffset] = 0.0f;
		//printf("\nImage:\n");
		for(int j=0; j<FIT_SIZE; j++){
			//int roffset = j * estimates.stride;
			int ioffset = (j+3) * image.stride;
			for(int i=0; i<FIT_SIZE; i++){
				//float residual = image.elements[index - FIT_RADIUS + i + ioffset] - estimates.elements[i + roffset];
				float residual = image.elements[index - FIT_RADIUS + i + ioffset];
				//printf("\n\n");
				for(int m=0; m<n; m++){
					//printf("Residual %d: %f\n", m, residual);
					xe[m + noffset] = xe[m + (n-1) * N_MAX];
					ye[m + noffset] = ye[m + (n-1) * N_MAX];
					mag[m + noffset] = mag[m + (n-1) * N_MAX];
					residual -= testMultiEvaluate(xe[m + noffset], ye[m + noffset], mag[m + noffset], index - FIT_RADIUS + i, j+3);
				}
				//printf(" | %f | ", image.elements[index - FIT_RADIUS + i + ioffset]);
				if(residual > mag[n + noffset]){
					mag[n + noffset] = residual;
					xe[n + noffset] = i + index - FIT_RADIUS;
					ye[n + noffset] = j + 3.0f;
				}
			}
			//printf("\n");
		}
		/*printf("\n\nxe[n + noffset]: %f, ye[n + noffset]: %f, mag[n + noffset]: %f",xe[n + noffset], ye[n + noffset], mag[n + noffset]);
		printf("\nResiduals:\n");
		for(int j=0; j<estimates.height; j++){
			int roffset = j * estimates.stride;
			int ioffset = (j+3) * image.stride;
			for(int i=0; i<estimates.width; i++){
				printf(" | %f | ", image.elements[index - FIT_RADIUS + i + ioffset] - estimates.elements[i + roffset]);
			}
			printf("\n");
		}*/
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

int factorial(int i){
	int ans = i--;
	while(i > 0){
		ans *= i;
		i--;
	}
	return ans;
}