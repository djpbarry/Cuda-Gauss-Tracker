#include <iterator>
#include <vector>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;

typedef vector<path> vec;

int countFiles(vector<path> v, char* ext);
vector<path> getFiles(char* folder);
float getInput(char* prompt, float default_val);
void getTextInput(char* prompt, char* result);

int countFiles(vector<path> v, char* ext){
	vector<path>::iterator v_iter;
	int frames = 0;
	for(v_iter = v.begin(); v_iter != v.end(); v_iter++){
		string ext_s = ((*v_iter).extension()).string();
		if((strcmp(ext_s.c_str(), ext) == 0)) {
			frames++;
		}
	}
	return frames;
}

vector<path> getFiles(char* folder)
{
  path p (folder);   // p reads clearer than argv[1] in the following code            // store paths,
  vec v;                                // so we can sort them later

  try
  {
    if (exists(p))    // does p actually exist?
    {
      if (is_directory(p))      // is p a directory?
      {
		printf("\nValid directory.\n");
        copy(directory_iterator(p), directory_iterator(), back_inserter(v));
        sort(v.begin(), v.end());             // sort, since directory iteration
                                              // is not ordered on some file systems
      }
	  else
		printf("\nInvalid directory.\n");
    }
    else
      printf("\nDirectory does not exist.\n");
  }

  catch (const filesystem_error& ex)
  {
    cout << ex.what() << '\n';
  }

  return v;
}

float getInput(char* prompt, float default_val){
	char inputs[INPUT_LENGTH];
	float result = default_val;
	printf("Enter %s (non-numeric for default): ", prompt);
	scanf_s("%9s", inputs, INPUT_LENGTH);
	float temp;
	if(sscanf_s(inputs, "%f", &temp) > 0){
		result = temp;
	}
	return result;
}

void getTextInput(char* prompt, char* result){
	char inputs[INPUT_LENGTH];
	printf("Enter %s (non-numeric for default): ", prompt);
	scanf_s("%9s", inputs, INPUT_LENGTH);
	char temp[INPUT_LENGTH];
	if(sscanf_s(inputs, "%9s", temp, INPUT_LENGTH) > 0){
		if(temp[0] != '.'){
			printf("\n%s doesn't look like a valid file extension, so I'm going to look for %s files\n", temp, result);
		} else {
			strcpy_s(result, INPUT_LENGTH * sizeof(char), temp);
		}
	}
	return;
}

int round(float number)
{
    return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);
}

char* toString(int i){
	int l = (int)floor(log10((float)i));
	l++;
	char *out = (char*) malloc (sizeof(char) * (l + 1));
	for(int j=l-1; j>=0; j--){
		out[j] = (char)(i % 10 + 48);
		i /= 10;
	}
	out[l] = '\0';
	return out;
}