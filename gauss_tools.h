#include <matrix_mat.h>

extern int findParticles(Mat image, Matrix B, int count, int frame, int fitRadius, float sigmaEst, float maxThresh, bool *warnings);
extern bool draw2DGaussian(Matrix image, float x0, float y0, float prec);
extern int maxFinder(const Matrix A, Matrix B, const float maxThresh, bool varyBG, int count, int k, int z, int fitRadius, bool *warnings);