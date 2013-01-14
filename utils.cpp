
#include <utils.h>
#include <matrix_mat.h>
#include <defs.h>

extern int countFiles(vector<path> v, char* ext) {
    vector<path>::iterator v_iter;
    int frames = 0;
    for (v_iter = v.begin(); v_iter != v.end(); v_iter++) {
        string ext_s = ((*v_iter).extension()).string();
        if ((strcmp(ext_s.c_str(), ext) == 0)) {
            frames++;
        }
    }
    return frames;
}

extern vector<path> getFiles(char* folder) {
    path p(folder); // p reads clearer than argv[1] in the following code            // store paths,
    vec v; // so we can sort them later

    try {
        if (exists(p)) // does p actually exist?
        {
            if (is_directory(p)) // is p a directory?
            {
                printf("\nValid directory.\n");
                copy(directory_iterator(p), directory_iterator(), back_inserter(v));
                sort(v.begin(), v.end()); // sort, since directory iteration
                // is not ordered on some file systems
            } else
                printf("\nInvalid directory.\n");
        } else
            printf("\nDirectory does not exist.\n");
    } catch (const filesystem_error& ex) {
        cout << ex.what() << '\n';
    }

    return v;
}

extern float getInput(char* prompt, float default_val) {
    char inputs[INPUT_LENGTH];
    float result = default_val;
    printf("Enter %s (non-numeric for default): ", prompt);
    scanf_s("%s", inputs, INPUT_LENGTH);
    float temp;
    if (sscanf_s(inputs, "%f", &temp) > 0) {
        result = temp;
    }
    return result;
}

extern void getTextInput(char* prompt, char* result) {
    char inputs[INPUT_LENGTH];
    printf("Enter %s: ", prompt);
    scanf_s("%s", inputs, INPUT_LENGTH);
    char temp[INPUT_LENGTH];
    if (sscanf_s(inputs, "%s", temp, INPUT_LENGTH) > 0) {
		strcpy_s(result, INPUT_LENGTH * sizeof (char), temp);
    }
    return;
}

extern int round(float number) {
    return (number >= 0) ? (int) (number + 0.5) : (int) (number - 0.5);
}

//Copies the width and height of the first image of extension ext found in the file list specified by v into dims.

extern void getDims(vector<path> v, const char* ext, int* dims) {
    vector<path>::iterator v_iter;
    for (v_iter = v.begin(); v_iter != v.end(); v_iter++) {
        string ext_s = ((*v_iter).extension()).string();
        if ((strcmp(ext_s.c_str(), ext) == 0)) {
            Mat frame = imread((*v_iter).string(), -1);
            dims[0] = frame.cols;
            dims[1] = frame.rows;
            return;
        }
    }
    return;
}

/*
	Load default parameter values from a configuration file.
*/
extern bool loadParams(float *params, int paramcount, char *filename, int maxline, char *inputFolder) {
    FILE *fp;
    FILE **fpp = &fp;
    fopen_s(fpp, filename, "r");
	char *line = (char*) malloc(maxline * sizeof(char));
	char *dummyString = (char*) malloc(maxline * sizeof(char));
	char dummyChar[1];
	int pindex = 0;
	if(fgets(line, maxline, fp) != NULL){
		sscanf_s(line, "%s %c %s", dummyString, maxline, dummyChar, sizeof(char), inputFolder, maxline);
	} else {
		return false;
	}
	while(fgets(line, maxline, fp) != NULL){
		sscanf_s(line, "%s %c %f", dummyString, maxline, dummyChar, sizeof(char), &params[pindex], sizeof(float));
		pindex++;
	}
    fclose(fp);
	if(pindex < paramcount){
		return false;
	} else {
		return true;
	}
}

int loadImages(Matrix destMatrix, char* ext, vector<path> v, char* folder, int numFiles, bool prompt) {
    //Load images into volume
    vector<path>::iterator v_iter;
    if(prompt) printf("\nLoading Images ... %d%%", 0);
    int thisFrame = 0;
    Mat frame;
    for (v_iter = v.begin(); v_iter != v.end(); v_iter++) {
        string ext_s = ((*v_iter).extension()).string();
        if ((strcmp(ext_s.c_str(), ext) == 0)) {
            if(prompt) printf("\rLoading Images ... %d%%", ((thisFrame + 1) * 100) / numFiles);
            frame = imread((*v_iter).string(), -1);
            copyToMatrix(frame, destMatrix, thisFrame);
            thisFrame++;
        }
    }
    return thisFrame;
}

void waitForKey(){
	getchar();
    getchar();
}