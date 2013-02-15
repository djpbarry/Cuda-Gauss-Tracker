
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
	vec v;
	if(folder == NULL){
		return v;
	}
    path p(folder);

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
    printf("%s", prompt);
    scanf_s("%s", inputs, INPUT_LENGTH);
    char temp[INPUT_LENGTH];
    if (sscanf_s(inputs, "%s", temp, INPUT_LENGTH) > 0) {
		strcpy_s(result, INPUT_LENGTH * sizeof (char), temp);
    }
    return;
}

extern bool getBoolInput(char* prompt) {
    printf("%s", prompt);
	char input = 10; // 10 = ''
	while(input == 10){
		input = getchar();
	}
    if (input == 'Y' || input == 'y'){
		return true;
	} else if (input == 'N' || input == 'n'){
		return false;
	} else {
		return getBoolInput(prompt);
	}
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
extern bool loadParams(float *params, int paramcount, char *filename, int maxline, char *inputFolder_c1, char *inputFolder_c2) {
    FILE *fp;
    FILE **fpp = &fp;
    fopen_s(fpp, filename, "r");
	char *line = (char*) malloc(maxline * sizeof(char));
	char *dummyString = (char*) malloc(maxline * sizeof(char));
	char dummyChar[1];
	int lineIndex = 1;
	if(fgets(line, maxline, fp) != NULL){
		if(sscanf_s(line, "%s %c %s", dummyString, maxline, dummyChar, sizeof(char), inputFolder_c1, maxline) != 3){
			memcpy(&inputFolder_c1[0], EMPTY, INPUT_LENGTH);
			_ASSERTE( _CrtCheckMemory( ) );
		}
	} else {
		return false;
	}
	if(fgets(line, maxline, fp) != NULL){
		if(sscanf_s(line, "%s %c %s", dummyString, maxline, dummyChar, sizeof(char), inputFolder_c2, maxline) != 3){
			memcpy(&inputFolder_c2[0], EMPTY, INPUT_LENGTH);
			_ASSERTE( _CrtCheckMemory( ) );
		}
	} else {
		return false;
	}
	while(fgets(line, maxline, fp) != NULL){
		sscanf_s(line, "%s %c %f", dummyString, maxline, dummyChar, sizeof(char), &params[lineIndex - 1], sizeof(float));
		lineIndex++;
	}
    fclose(fp);
	if(lineIndex != paramcount ){
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

extern void waitForKey(){
	getchar();
    getchar();
}

void getParams(float* _spatialRes, float* _numAp, float* _lambda, float* _sigmaEstNM, float* _sigmaEstPix, int* _scalefactor, float* _maxThresh, char* _ext,
	char* folder_c1, char* folder_c2, char* file, bool* verbose){
	float params[NUM_PARAMS];
	char tempExt[INPUT_LENGTH];
	if(loadParams(params, NUM_PARAMS, file, INPUT_LENGTH, folder_c1, folder_c2)){
		*_spatialRes = params[0];
		*_numAp = params[1];
		*_lambda = params[2];
		*_sigmaEstNM = params[3] * *_lambda / (*_numAp * *_spatialRes);
		*_sigmaEstPix = params[3] * *_lambda / *_numAp;
		*_scalefactor = (int)round(params[4]);
		*_maxThresh = params[5];
	} else {
		printf("Failed to load configuration file: %s\n\n", file);
	}
	*_spatialRes = getInput("spatial resolution in nm", *_spatialRes);
    *_lambda = getInput("wavelength of emitted light in nm", *_lambda);
	*_maxThresh = getInput("maximum intensity threshold", *_maxThresh);
    *_scalefactor = round(getInput("scaling factor for output", (float) *_scalefactor));
    getTextInput("Enter file extension: ", tempExt);
	*verbose = getBoolInput("Verbose mode (Y/N)?");
	if (tempExt[0] != '.') {
		printf("\n%s doesn't look like a valid file extension, so I'm going to look for %s files\n", tempExt, _ext);
    } else {
		strcpy_s(_ext, INPUT_LENGTH * sizeof (char), tempExt);
	}
}

int getCurrentRevisionNumber(char* filename, int maxline){
	int revision;
	FILE *fp;
    FILE **fpp = &fp;
    fopen_s(fpp, filename, "r");
	char *line = (char*) malloc(maxline * sizeof(char));
	char *dummyString = (char*) malloc(maxline * sizeof(char));
	if(fgets(line, maxline, fp) != NULL){
		sscanf_s(line, "%d %s", &revision, dummyString, maxline);
	} else {
		return -1;
	}
    fclose(fp);
	return revision;
}

/*
* Ensure directory is formatted according to UNIX conventions
*/
extern void checkFileSep(char* directory){
	int i = 0;
	while(directory[i] != '\0'){
		if(directory[i] == '\\'){
			directory[i] = '/';
		}
		i++;
	}
	return;
}