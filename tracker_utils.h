#ifndef _TRACKER_UTILS_
#define _TRACKER_UTILS_

#include "stdafx.h"
#include <cv.h>
#include <highgui.h>

using namespace cv;

extern void output(int* dims, int frames, string outputDir, int _mNbParticles, int* _counts, float* _mParticlesMemory, int _scalefactor);

#endif