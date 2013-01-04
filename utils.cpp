
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
