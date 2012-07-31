typedef struct { 
    int width; 
    int height;
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