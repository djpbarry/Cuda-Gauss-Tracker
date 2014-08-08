#ifndef _CUDA_GAUSS_FITTER_
#define _CUDA_GAUSS_FITTER_

#include <matrix.h>

extern "C" float GaussFitter(Matrix A, int maxcount, float sigEst);

#endif