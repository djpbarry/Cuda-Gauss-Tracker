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

extern int countFiles(vector<path> v, char* ext);
extern vector<path> getFiles(char* folder);
extern float getInput(char* prompt, float default_val);
extern void getTextInput(char* prompt, char* result);
extern int round(float number);
extern void getDims(vector<path> v, const char* ext, int* dims);

#endif