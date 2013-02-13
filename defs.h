#ifndef _DEFS_
#define _DEFS_

//Format of vector for tracker = {_X_, _Y_, _Z_, _X_VEL_, _Y_VEL_, _Z_VEL_, _MAG_}
#define _X_ 0
#define _Y_ 1
#define _Z_ 2
#define _X_VEL_ 3
#define _Y_VEL_ 4
#define _Z_VEL_ 5
#define _MAG_ 6

#define NUM_PARAMS 7
#define N_MAX 3
#define MAX_DETECTIONS 200
#define FIT_RADIUS 3
#define FIT_SIZE 7
#define SEARCH_RADIUS 1
#define STEP_TOL 2000
#define ITERATIONS 100
#define XY_STEP_SIZE 0.1f
#define MAG_STEP_SIZE 10.0f
#define BG_STEP_SIZE 1.0f
#define BLOCK_SIZE_X 64
#define BLOCK_SIZE_Y 1
#define HEADER 3
#define FOOTER (4*N_MAX + 1)
#define DATA_ROWS (HEADER + FOOTER)
#define X_ROW 0
#define Y_ROW 1
#define Z_ROW 2
#define BEST_ROW (Z_ROW + FIT_SIZE + 1)
#define XE_ROW (BEST_ROW + 1)
#define YE_ROW (XE_ROW + N_MAX)
#define MAG_ROW (YE_ROW + N_MAX)
#define BG_ROW (MAG_ROW + N_MAX)
#define PNG ".png"
#define INPUT_LENGTH 200
#define DIM_OF_STATE 7
#define BACKGROUND 1.0f
#define TIF ".tif"
#define MAX_MEMORY 500000000

#endif