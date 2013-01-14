#include <global_params.h>

extern float _spatialRes = 40.3125f;
extern float _numAp = 1.4f;
extern float _lambda = 561.0f;
extern float _sigmaEstNM = 0.305f * _lambda / (_numAp * _spatialRes);
extern int _scalefactor = 1;
extern float _mSigmaPSFxy = 0.305f * _lambda / _numAp;