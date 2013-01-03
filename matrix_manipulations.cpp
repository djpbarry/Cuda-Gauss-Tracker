#include <matrix_mat.h>
#include <stdio.h>

extern void saveMatrix(Matrix source, int x, int y, int radius) {
    FILE *fp;
    FILE **fpp = &fp;
    fopen_s(fpp, "C:/users/barry05/Desktop/matrix.txt", "w");
    for (int m = y - radius; m <= y + radius; m++) {
        int offset = m * source.stride;
        for (int n = x - radius; n <= x + radius; n++) {
            int bx = n - x + radius;
            fprintf(fp, "%.0f ", source.elements[offset + n]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/*
 * Copies the contents of Matrix source into dest. Elements within source are copied starting at the specified index and stopping when dest is full.
 * @param source The source Matrix to be copied
 * @param dest The destination Matrix
 * @param start The first index of source elements to be copied
 */
extern void matrixCopy(Matrix source, Matrix dest, int start) {
    for (int i = 0; i < dest.size; i++) {
        dest.elements[i] = source.elements[i + start];
    }
}
