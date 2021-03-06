#ifndef _GAUSS_TOOLS_
#define _GAUSS_TOOLS_

#include <cv.h>
#include <highgui.h>
#include <matrix.h>

using namespace cv;

extern int findParticles(Mat image, Matrix B, int count, int frame, int fitRadius, float sigmaEst, float maxThresh, bool *warnings, bool copyRegions);
extern bool draw2DGaussian(Matrix image, float x0, float y0, float prec);
extern int maxFinder(int* point, const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z, int fitRadius, bool *warnings, bool copyRegions);
extern bool drawDot(Matrix image, float x0, float y0);

#endif