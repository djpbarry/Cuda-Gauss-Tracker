#ifndef _DEFS_
#define _DEFS_

#define FILENAME_LENGTH 10
#define N_MAX 5
#define MAX_DETECTIONS 1000
#define FIT_RADIUS 3
#define FIT_SIZE 7
#define SEARCH_RADIUS 1
#define STEP_TOL 2000
#define ITERATIONS 100
#define XY_STEP_SIZE 0.1f
#define MAG_STEP_SIZE 10.0f
#define BLOCK_SIZE_X 256
#define BLOCK_SIZE_Y 1
#define HEADER 3
#define FOOTER (3*N_MAX + 1)
#define DATA_ROWS (HEADER + FOOTER)
#define X_ROW 0
#define Y_ROW 1
#define Z_ROW 2
#define BEST_ROW (Z_ROW + FIT_SIZE + 1)
#define XE_ROW (BEST_ROW + 1)
#define YE_ROW (XE_ROW + N_MAX)
#define MAG_ROW (YE_ROW + N_MAX)
#define PNG ".png"
#define INPUT_LENGTH 10
#define DIM_OF_STATE 7

#endif