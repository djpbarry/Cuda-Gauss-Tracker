#ifndef _MATRIX_MAT_
#define _MATRIX_MAT_

#include <cv.h>
#include <highgui.h>
#include <matrix.h>

using namespace cv;

extern void copyFromMatrix(Mat M, Matrix A, int index, float scale);
extern void copyToMatrix(Mat M, Matrix A, int index);

#endif