#include <drawing.h>
#include <math.h>

/*
	Draw a dot in image at the specified (x0, y0) coordinate.
*/
void drawDot(Matrix image, int x0, int y0) {
	int index = x0 + y0 * image.stride;
    image.elements[index] = image.elements[index] + 1.0f;
    return;
}

/*
	Draw a square of side length (2 * radius + 1) in image at the specified (x0, y0) coordinate.
*/
bool drawSquare(Matrix image, float x0, float y0, int drawRadius) {
    int x, y;
    for (x = (int) floor(x0 - drawRadius); x <= x0 + drawRadius; x++) {
        y = (int) floor(y0 - drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
        y = (int) floor(y0 + drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
    }

    for (y = (int) floor(y0 - drawRadius); y <= y0 + drawRadius; y++) {
        x = (int) floor(x0 - drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
        x = (int) floor(x0 + drawRadius);
        if (x >= 0 && x < image.width && y >= 0 && y < image.height) {
            int index = x + y * image.stride;
            image.elements[index] = 255.0f;
        }
    }
    return true;
}
