
#include "stdafx.h"
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <matrix.h>

using namespace cv;

int findParticles(Mat image, Matrix B, int count, int z);

extern "C" float GaussFitter(Matrix A, int maxcount, float sigEst, float maxThresh);

void copyToMatrix(Mat, Matrix);

void copyFromMatrix(Mat, Matrix);

char* toString(int);

int round(float);

int testMaxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int z);

float testEvaluate(float x0, float y0, float max, int x, int y, float sig2);

float testGetRSquared(int x0, float srs, Matrix M);

bool testDraw2DGaussian(Matrix image, float mag, float x0, float y0);

void testDoMultiFit(Matrix M, int x0, int N, float *xe, float *ye, float *mag, float *r);

float testSumMultiResiduals(int x0, float *xe, float *ye, float *mag, Matrix M, float xinc, float yinc, float minc, int index, int N);

float testMultiEvaluate(float x0, float y0, float mag, int x, int y);

int testInitialiseFitting(Matrix image, int index, int N, float *xe, float *ye, float *mag);

void testCentreOfMass(float *x, float *y, int index, Matrix image);

float _spatialRes = 133.0f;
float _sigmaEst, _2sig2;
float _maxThresh = 2.0f;
float _numAp = 1.4f;
float _lambda = 488.0f;
int _numFrames = 299;
int _scalefactor = 4;

int main(int argc, char* argv[])
{
	// Sigma estimate for Gaussian fitting
	_sigmaEst = 0.305f * _lambda / (_numAp * _spatialRes);
	_2sig2 = 2.0f * _sigmaEst * _sigmaEst;

	printf("Start...\n");
	String folder = "C:\\Users\\barry05\\Desktop\\CUDA Gauss Localiser Tests\\Test6\\";
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
	printf("\nFinding Maxima ... %d", 0);
	while(!frame.empty()){
		count = findParticles(frame, candidates, count, frames);
		frames++;
		filename = strcat(toString(frames),TIF);
		float noughts = log10((float)frames);
		while(noughts < 2){
			filename = '0' + filename;
			noughts += 1.0f;
		}
		frame = imread(folder + filename, -1);
		printf("\rFinding Maxima ... %d", frames);
	}
	int outcount = 0;
	savefilename = "000.tif";
	printf("\n\nGaussian Fitting\n-------------------------");
	GaussFitter(candidates, count, _sigmaEst, _maxThresh);
	clock_t totaltime = 0;
	printf("\nWriting Output ... %d", 0);
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
		cudaoutput.width = width*_scalefactor;
		cudaoutput.stride=cudaoutput.width;
		cudaoutput.height = height*_scalefactor;
		cudaoutput.size = cudaoutput.width * cudaoutput.height;
		cudaoutput.elements = (float*)malloc(sizeof(float) * cudaoutput.size);
		for(int p=0; p<cudaoutput.width*cudaoutput.height; p++){
			cudaoutput.elements[p] = 0.0f;
		}
		Mat cudasaveframe(height*_scalefactor,width*_scalefactor,CV_32F);
/*		Matrix testoutput;
		testoutput.width = width*_scalefactor;
		testoutput.stride=testoutput.width;
		testoutput.height = height*_scalefactor;
		testoutput.size = cudaoutput.width * cudaoutput.height;
		testoutput.elements = (float*)malloc(sizeof(float) * testoutput.size);
		for(int p=0; p<width*height*_scalefactor*_scalefactor; p++){
			testoutput.elements[p] = 0.0f;
		}
		Mat testsaveframe(height*_scalefactor,width*_scalefactor,CV_32F);*/
		while(round(candidates.elements[outcount + candidates.stride * Z_ROW]) <= z && outcount<count){
			int x = round(candidates.elements[outcount + candidates.stride * X_ROW]);
			int y = round(candidates.elements[outcount + candidates.stride * Y_ROW]);
			int best = round(candidates.elements[outcount + candidates.stride * BEST_ROW]);
			int xRegionCentre = outcount*FIT_SIZE+FIT_RADIUS;
			int yRegionCentre = FIT_RADIUS+HEADER;
			if(best>=0){
				for(int i=0; i<=best; i++){
					//printf("\nxe: %f ye: %f mag: %f\n",x + candidates.elements[outcount + candidates.stride * (XE_ROW + i)] - xRegionCentre,y + candidates.elements[outcount + candidates.stride * (YE_ROW + i)] - yRegionCentre,candidates.elements[outcount + candidates.stride * (MAG_ROW + i)]);
					if(candidates.elements[outcount + candidates.stride * (MAG_ROW + i)] > _maxThresh){
						testDraw2DGaussian(cudaoutput, candidates.elements[outcount + candidates.stride * (MAG_ROW + i)],
						x + candidates.elements[outcount + candidates.stride * (XE_ROW + i)] - xRegionCentre,
						y + candidates.elements[outcount + candidates.stride * (YE_ROW + i)] - yRegionCentre);
					}
				}
			}
			/*float *xe = (float*)malloc(sizeof(float) * N_MAX * N_MAX);
			float *ye = (float*)malloc(sizeof(float) * N_MAX * N_MAX);
			float *mag = (float*)malloc(sizeof(float) * N_MAX * N_MAX);
			clock_t start = clock();
			best = testInitialiseFitting(candidates, xRegionCentre, N_MAX, xe, ye, mag);
			totaltime += clock() - start;
			if(best>=0){
				for(int i=0; i<=best; i++){
					testDraw2DGaussian(testoutput, mag[best * N_MAX + i], x + xe[best * N_MAX + i] - xRegionCentre, y + ye[best * N_MAX + i] - yRegionCentre);
				}
			}*/
			outcount++;
		}
		copyFromMatrix(cudasaveframe, cudaoutput);
		cudasaveframe.convertTo(cudasaveframe,CV_16UC1);
		printf("\rWriting Output ... %d", z);
		imwrite(folder + "CudaOutput\\" + savefilename, cudasaveframe);
		/*copyFromMatrix(testsaveframe, testoutput);
		imwrite(folder + "TestOutput\\" + savefilename, testsaveframe);*/
		free(cudaoutput.elements);
	}
	char c;
	//printf("\n\nReference Time: %.0f", totaltime * 1000.0f/CLOCKS_PER_SEC);
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
	float scale = 65535.0f / 255.0f;
	for(int y=0; y< M.rows; y++){
		int Moffset = y * M.step1();
		int Aoffset = y * A.stride;
		for(int x=0; x < M.cols; x++){
			float a = A.elements[Aoffset + x];
			((float*)M.data)[Moffset + x] = scale * A.elements[Aoffset + x];
			//if(((float*)M.data)[Moffset + x] > 0.0f) printf("Val: %f\n", ((float*)M.data)[Moffset + x]);
		}
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
       // int x, y;
		float x0 = x01 * _scalefactor;
		float y0 = y01 * _scalefactor;
		//int drawRadius = FIT_RADIUS + 2;
       // for (x = (int) floor(x0 - drawRadius); x <= x0 + drawRadius; x++) {
          //  for (y = (int) floor(y0 - drawRadius); y <= y0 + drawRadius; y++) {
                /* The current pixel value is added so as not to "overwrite" other
                Gaussians in close proximity: */
				//if(x >= 0 && x < image.width && y >= 0 && y < image.height){
					//image.elements[x + y * image.stride] = image.elements[x + y * image.stride] + testMultiEvaluate(x0, y0, mag, x, y);
					int index = (int) floor(x0) + (int) floor(y0) * image.stride;
					image.elements[index] = image.elements[index] + 1.0f;
				//}
        //    }
   //     }
        return true;
    }

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
