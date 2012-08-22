#ifndef _UTILS_
#define _UTILS_

#include <iterator>
#include <vector>
#include <boost/filesystem.hpp>
#include <matrix.h>
#include <cv.h>
#include <highgui.h>
#include <defs.h>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

typedef vector<path> vec;

extern int countFiles(vector<path> v, char* ext);
extern vector<path> getFiles(char* folder);
extern float getInput(char* prompt, float default_val);
extern void getTextInput(char* prompt, char* result);
extern int round(float number);
extern void copyToMatrix(Mat M, Matrix A, int index);
extern void copyFromMatrix(Mat M, Matrix A, int index, float scale);
extern void getDims(vector<path> v, const char* ext, int* dims);
extern void matrixCopy(Matrix source, Matrix dest, int start);

extern  int countFiles(vector<path> v, char* ext){
	vector<path>::iterator v_iter;
	int frames = 0;
	for(v_iter = v.begin(); v_iter != v.end(); v_iter++){
		string ext_s = ((*v_iter).extension()).string();
		if((strcmp(ext_s.c_str(), ext) == 0)) {
			frames++;
		}
	}
	return frames;
}

extern  vector<path> getFiles(char* folder)
{
  path p (folder);   // p reads clearer than argv[1] in the following code            // store paths,
  vec v;                                // so we can sort them later

  try
  {
    if (exists(p))    // does p actually exist?
    {
      if (is_directory(p))      // is p a directory?
      {
		printf("\nValid directory.\n");
        copy(directory_iterator(p), directory_iterator(), back_inserter(v));
        sort(v.begin(), v.end());             // sort, since directory iteration
                                              // is not ordered on some file systems
      }
	  else
		printf("\nInvalid directory.\n");
    }
    else
      printf("\nDirectory does not exist.\n");
  }

  catch (const filesystem_error& ex)
  {
    cout << ex.what() << '\n';
  }

  return v;
}

extern  float getInput(char* prompt, float default_val){
	char inputs[INPUT_LENGTH];
	float result = default_val;
	printf("Enter %s (non-numeric for default): ", prompt);
	scanf_s("%9s", inputs, INPUT_LENGTH);
	float temp;
	if(sscanf_s(inputs, "%f", &temp) > 0){
		result = temp;
	}
	return result;
}

extern  void getTextInput(char* prompt, char* result){
	char inputs[INPUT_LENGTH];
	printf("Enter %s (non-numeric for default): ", prompt);
	scanf_s("%9s", inputs, INPUT_LENGTH);
	char temp[INPUT_LENGTH];
	if(sscanf_s(inputs, "%9s", temp, INPUT_LENGTH) > 0){
		if(temp[0] != '.'){
			printf("\n%s doesn't look like a valid file extension, so I'm going to look for %s files\n", temp, result);
		} else {
			strcpy_s(result, INPUT_LENGTH * sizeof(char), temp);
		}
	}
	return;
}

extern  int round(float number)
{
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}

//Copy elements from an OpenCV Mat structure into a Matrix. Necessary as CUDA and OpenCV inter-operability is poor.
extern  void copyToMatrix(Mat M, Matrix A, int index){
	int frameoffset = index * A.width * A.height;
	for(int y=0; y< M.rows; y++){
		int Moffset = y * M.step1();
		int Aoffset = y * A.stride + frameoffset;
		for(int x=0; x < M.cols; x++){
			A.elements[Aoffset + x] = ((float*)M.data)[Moffset + x];
		}
	}
	return;
}

//Copy elements from a Matrix into an OpenCV Mat structure. Necessary as CUDA and OpenCV inter-operability is poor.
extern  void copyFromMatrix(Mat M, Matrix A, int index, float scale){
	int frameoffset = index * A.width * A.height;
	for(int y=0; y< M.rows; y++){
		int Moffset = y * M.step1();
		int Aoffset = y * A.stride + frameoffset;
		for(int x=0; x < M.cols; x++){
			((float*)M.data)[Moffset + x] = scale * A.elements[Aoffset + x];
		}
	}
	return;
}

//Copies the width and height of the first image of extension ext found in the file list specified by v into dims.
extern  void getDims(vector<path> v, const char* ext, int* dims){
	vector<path>::iterator v_iter;
	for(v_iter = v.begin(); v_iter != v.end(); v_iter++){
		string ext_s = ((*v_iter).extension()).string();
		if((strcmp(ext_s.c_str(), ext) == 0)) {
			Mat frame = imread((*v_iter).string(), -1);
			dims[0] = frame.cols;
			dims[1] = frame.rows;
			return;
		}
	}
	return;
}

/*
* Copies the contents of Matrix source into dest. Elements within source are copied starting at the specified index and stopping when dest is full.
* @param source The source Matrix to be copied
* @param dest The destination Matrix
* @param start The first index of source elements to be copied
*/
extern  void matrixCopy(Matrix source, Matrix dest, int start){
	for(int i=0; i < dest.size; i++){
		dest.elements[i] = source.elements[i + start];
	}
}

#endif