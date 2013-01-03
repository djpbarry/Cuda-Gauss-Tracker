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

extern void saveMatrix(Matrix source, int x, int y, int radius);
extern void matrixCopy(Matrix source, Matrix dest, int start);

#endif