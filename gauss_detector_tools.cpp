
#include <gauss_tools.h>
#include <matrix_mat.h>
#include <defs.h>
#include <boost/lexical_cast.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/round.hpp>

float evaluate(float x0, float y0, int x, int y, float sigma2);
float getFitPrecision(Matrix candidates, int index, int x0, int y0, int best, int bgIndex, int magIndex, float spatialRes, float sigmaEst);

extern int findParticles(Mat image, Matrix B, int count, int frame, int fitRadius, float sigmaEst, float maxThresh, bool *warnings, bool copyRegions) {
    Size radius(2 * fitRadius + 1, 2 * fitRadius + 1);
    Mat temp = image.clone();
    GaussianBlur(temp, image, radius, sigmaEst, sigmaEst);
    Matrix A;
    A.width = image.cols;
    A.stride = A.width;
    A.height = image.rows;
    A.size = A.width * A.height;
    A.elements = (float*) malloc(sizeof (float) * A.size);
    copyToMatrix(temp, A, 0);
    int thisCount = maxFinder(NULL, A, B, maxThresh, true, count, 0, frame, fitRadius, warnings, copyRegions);
    free(A.elements);
    return thisCount;
}

/*
	Searches for local maxima in A, greater in magnitude than maxThresh, and copies the local neighbourhood
	surrounding the maximum into B. Returns the total number of detected maxima in A.
*/

extern int maxFinder(int* point, const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z, int fitRadius, bool *warnings, bool copyRegions) {
    float min, max;
    int i, j;
    int koffset = k * A.width * A.height;
	int fitSize = 2 * fitRadius + 1;
	int x1 = fitRadius, x2 = A.width - fitRadius, y1 = fitRadius, y2 = A.height - fitRadius;
	if(point != NULL){
		x1 = (point[0] - fitRadius < fitRadius) ? fitRadius : point[0] - fitRadius;
		x2 = (point[0] + fitRadius + 1 > A.width) ? A.width: point[0] + fitRadius + 1;
		y1 = (point[1] - fitRadius < fitRadius) ? fitRadius : point[1] - fitRadius;
		y2 = (point[1] + fitRadius + 1 > A.height) ? A.height: point[1] + fitRadius + 1;
	}
    for (int y = y1; y < y2; y++) {
        for (int x = x1; x < x2; x++) {
			for (min = FLT_MAX, max = -FLT_MAX, j = y - fitRadius; j <= y + fitRadius; j++) {
                int offset = koffset + j * A.stride;
				for (i = x - fitRadius; i <= x + fitRadius; i++) {
                    float pix = A.elements[i + offset];
                    warnings[1] = warnings[1] || (pix > 127.0f);
                    warnings[0] = warnings[0] && (pix < 1.0f);
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
				if(copyRegions){
					int bxoffset = fitSize * count;
					for (int m = y - fitRadius; m <= y + fitRadius; m++) {
						int aoffset = m * A.stride;
						int boffset = (m - y + fitRadius + 3) * B.stride;
						for (int n = x - fitRadius; n <= x + fitRadius; n++) {
							int bx = n - x + fitRadius;
							B.elements[boffset + bx + bxoffset] = A.elements[aoffset + n];
						}
					}
				}
				if(B.elements != NULL){
					B.elements[count] = (float) x;
					B.elements[count + B.stride] = (float) y;
					B.elements[count + 2 * B.stride] = (float) z;
				}
                count++;
				if(count >= MAX_DETECTIONS) return count;
            }
        }
    }
    return count;
}

float evaluate(float x0, float y0, int x, int y, float sigma2) {
    return (1.0f / (2.0f * boost::math::constants::pi<float>() * sigma2)) * exp(-((x - x0)*(x - x0) + (y - y0)*(y - y0)) / (sigma2));
}

/*
	Draw a normalised 2D Gaussian distribution in image, centred at (x0, y0), with the width determined by the localisation precision, prec
	(0 < prec < 1).
*/
extern bool draw2DGaussian(Matrix image, float x0, float y0, float prec) {
    int x, y;
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

extern bool drawDot(Matrix image, float x0, float y0) {
    int x = (int) floor(x0);
	int y = (int) floor(y0);
    if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
        int index = x + y * image.stride;
        image.elements[index] = image.elements[index] + 1.0f;
    }
    return true;
}

/*
	Estimates the precision of the Gaussian fits in candidates, based on the number of photons collected (the height of the peak)
	and the level of background.
*/
float getFitPrecision(Matrix candidates, int index, int x0, int y0, int best, int bgIndex, int magIndex, float spatialRes, float sigmaEst) {
    //int y0 = HEADER;
    //int x0 = index * FIT_SIZE;
    //int best = round(candidates.elements[index + candidates.stride * BEST_ROW]);
    float bg = 0.0f;
    float N = 0.0f;
	float sig2 = sigmaEst * sigmaEst;
	float res2 = spatialRes * spatialRes;
    for (int k = 0; k <= best; k++) {
        bg += candidates.elements[index + candidates.stride * (bgIndex + k)];
        float mag = candidates.elements[index + candidates.stride * (magIndex + k)];
        N += mag * (2.0f * boost::math::constants::pi<float>() * sig2);
    }
    if (N > 0.0f) {
        bg /= (best + 1);
        float s = 2.0f * sigmaEst * spatialRes;
        float a = (s * s) + (res2 / 12.0f);
        float b = 8.0f * (boost::math::constants::pi<float>()) * pow(s, 4.0f) * bg * bg;
        return pow((a / N) + (b / (res2 * N * N)), 0.5f);
    } else {
        return -1.0f;
    }
}
