#ifndef _MATRIX_
#define _MATRIX_

typedef struct { 
    int width; 
    int height;
	int depth;
	int size;
	int stride;
    float* elements; 
} Matrix;

typedef struct { 
    int width; 
    int height;
	int stride;
    char* elements; 
} charMatrix;

#endif