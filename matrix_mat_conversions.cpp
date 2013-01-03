#include <matrix_mat.h>

//Copy elements from a Matrix into an OpenCV Mat structure. Necessary as CUDA and OpenCV inter-operability is poor.

extern void copyFromMatrix(Mat M, Matrix A, int index, float scale) {
    int frameoffset = index * A.width * A.height;
    for (int y = 0; y < M.rows; y++) {
        int Moffset = y * M.step1();
        int Aoffset = y * A.stride + frameoffset;
        for (int x = 0; x < M.cols; x++) {
            ((float*) M.data)[Moffset + x] = scale * A.elements[Aoffset + x];
        }
    }
    return;
}

//Copy elements from an OpenCV Mat structure into a Matrix. Necessary as CUDA and OpenCV inter-operability is poor.

extern void copyToMatrix(Mat M, Matrix A, int index) {
    int frameoffset = index * A.width * A.height;
    for (int y = 0; y < M.rows; y++) {
        int Moffset = y * M.step1();
        int Aoffset = y * A.stride + frameoffset;
        for (int x = 0; x < M.cols; x++) {
            A.elements[Aoffset + x] = ((float*) M.data)[Moffset + x];
        }
    }
    return;
}