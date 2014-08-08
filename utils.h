#ifndef _UTILS_
#define _UTILS_

#include <iterator>
#include <vector>
#include <boost/filesystem.hpp>
#include <matrix.h>
#include <cv.h>
#include <highgui.h>

using namespace cv;
using namespace std;
using namespace boost::filesystem;

typedef vector<path> vec;

extern int countFiles(vector<path> v, const char* ext);
extern vector<path> getFiles(const char* folder);
extern float getInput(char* prompt, float default_val);
extern void getTextInput(char* prompt, char* result);
extern bool getBoolInput(char* prompt);
extern int round(float number);
extern void getDims(vector<path> v, const char* ext, int* dims);
extern bool loadParams(float *params, int paramcount, char *filename, int maxline, char *inputFolder);
extern int loadImages(Matrix destMatrix, char* ext, vector<path> v, char* folder, int numFiles, bool prompt);
extern void waitForKey();
extern void getParams(float* _spatialRes, float* _numAp, float* _lambda, float* _sigmaEstNM, float* _sigmaEstPix, int* _scalefactor, float* _maxThresh, char* _ext, char* folder_c1, char* folder_c2, char* file, bool* verbose);
extern int getCurrentRevisionNumber(char* filename, int maxline);
extern void checkFileSep(char* directory);
extern float getPercentileThresh(const Mat *image, float percentThresh);

#endif