#include <tracker_tools.h>

float _mSigmaOfRandomWalk[] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
float _mSigmaOfDynamics[] = {500.0f, 500.0f, 1.0f, 1.0f};
int _mResamplingThreshold = 250;
int _mRepSteps = 5;

normal_distribution<float> _dist(0.0f, 1.0f);
mt19937 rng;
variate_generator<mt19937, normal_distribution<float> > var_nor(rng, _dist);

void createParticles(float* aStateVectors, float* aParticles, int _currentLength, int _mNbParticles) {
    printf("\nCreating Particles ... %d%%", 0);
    for (int i = 0; i < _currentLength; i++) {
        printf("\rCreating Particles ... %d%%", ((i + 1) * 100) / _currentLength);
        int stateVectorIndex = i * DIM_OF_STATE;
        int stateVectorParticleIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        for (int vIndex = 0; vIndex < _mNbParticles; vIndex++) {
            float vProposal[DIM_OF_STATE + 1];
            int particleIndex = stateVectorParticleIndex + vIndex * (DIM_OF_STATE + 1);
            for (int vI = 0; vI < DIM_OF_STATE; vI++) {
                vProposal[vI] = aStateVectors[stateVectorIndex + vI];
            }
            //				Init the weight as a last dimension
            vProposal[DIM_OF_STATE] = 1.0f; //not 0!			
            //add the new particle
            copyParticle(aParticles, vProposal, particleIndex);
        }
    }
    return;
}

void filterTheInitialization(Matrix aImageStack, int aInitPFIterations, float* _mParticles,	int _currentLength, int _mNbParticles, float* _mStateVectors) {
    printf("\nInitialising...");
    float vSigmaOfRWSave[DIM_OF_STATE];
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        vSigmaOfRWSave[vI] = _mSigmaOfRandomWalk[vI];
    }
    Matrix frame;
    frame.width = aImageStack.width;
    frame.height = aImageStack.height;
    frame.stride = frame.width;
    frame.depth = 1;
    frame.size = frame.width * frame.height;
    frame.elements = (float*) malloc(sizeof (float) * frame.size);
    matrixCopy(aImageStack, frame, 0);

    for (int vR = 0; vR < aInitPFIterations; vR++) {
        scaleSigmaOfRW(1.0f / powf(3.0f, (float) vR));
        DrawParticlesWithRW(_mParticles, _currentLength, _mNbParticles);

        updateParticleWeightsOnGPU(frame, _mParticles, _currentLength, _mNbParticles);
        //updateParticleWeights(aImageStack, aFrameOfInit);

        estimateStateVectors(_mStateVectors, _mParticles, _currentLength, _mNbParticles);

        resample(_mParticles, _currentLength, _mNbParticles);
    }

    //restore the sigma vector
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        _mSigmaOfRandomWalk[vI] = vSigmaOfRWSave[vI];
    }
    free(frame.elements);
    return;
}

void scaleSigmaOfRW(float vScaler) {
    for (int vI = 0; vI < DIM_OF_STATE; vI++) {
        _mSigmaOfRandomWalk[vI] *= vScaler;
    }
}

void DrawParticlesWithRW(float* aParticles, int _currentLength, int _mNbParticles) {
    //printf("\nDrawing Particles ... %d%%", 0);
    for (int i = 0; i < _currentLength; i++) {
        //printf("\rDrawing Particles ... %d%%", ((i + 1) * 100) / _currentLength);
        int stateVectorIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        for (int j = 0; j < _mNbParticles; j++) {
            int particleIndex = stateVectorIndex + j * (DIM_OF_STATE + 1);
            randomWalkProposal(aParticles, particleIndex);
        }
    }
    return;
}

void randomWalkProposal(float* aParticles, int offset) {
    for (int aI = 0; aI < DIM_OF_STATE; aI++) {
        aParticles[aI + offset] += var_nor() * _mSigmaOfRandomWalk[aI];
    }
    return;
}

/*void updateParticleWeights(Matrix aObservationStack, int aFrameIndex, int _currentLength, int _mNbParticles, float* _mParticles, float* _mMaxLogLikelihood) {
    //printf("\nUpdating Weights ... %d%%", 0);
    for (int i = 0; i < _currentLength; i++) {
        //printf("\rUpdating Weights ... %d%%", ((i + 1) * 100) / _currentLength);
        int stateVectorIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        float vSumOfWeights = 0.0f;
        //
        // Calculate the likelihoods for each particle and save the biggest one
        //
        float* vLogLikelihoods = (float*) malloc(sizeof (float) * _mNbParticles);
        float vMaxLogLikelihood = -FLT_MAX;
        bool* vBitmap = (bool*)malloc(sizeof (bool) * aObservationStack.width * aObservationStack.height);
        generateParticlesIntensityBitmap(_mParticles, stateVectorIndex, vBitmap, aObservationStack.width, aObservationStack.height, _mNbParticles);
        calculateLikelihoods(aObservationStack, vBitmap, aFrameIndex, vLogLikelihoods, _mParticles, stateVectorIndex, _mNbParticles);
        for (int vI = 0; vI < _mNbParticles; vI++) {
            if (vLogLikelihoods[vI] > vMaxLogLikelihood) {
                vMaxLogLikelihood = vLogLikelihoods[vI];
            }
        }
        _mMaxLogLikelihood[aFrameIndex - 1] = vMaxLogLikelihood;
        //
        // Iterate again and update the weights
        //
        int vI = 0;
        for (int vI = 0; vI < _mNbParticles; vI++) {
            vLogLikelihoods[vI] -= vMaxLogLikelihood;
            int particleWeightIndex = stateVectorIndex + vI * (DIM_OF_STATE + 1) + DIM_OF_STATE;
            _mParticles[particleWeightIndex] = _mParticles[particleWeightIndex] * expf(vLogLikelihoods[vI]);
            vSumOfWeights += _mParticles[particleWeightIndex];
        }
        //
        // Iterate again and normalize the weights
        //
        if (vSumOfWeights == 0.0f) { //can happen if the winning particle before had a weight of 0.0
            for (int vI = 0; vI < _mNbParticles; vI++) {
                _mParticles[stateVectorIndex + vI * (DIM_OF_STATE + 1) + DIM_OF_STATE] = 1.0f / (float) _mNbParticles;
            }
        } else {
            for (int vI = 0; vI < _mNbParticles; vI++) {
                _mParticles[stateVectorIndex + vI * (DIM_OF_STATE + 1) + DIM_OF_STATE] /= vSumOfWeights;
            }
        }
        free(vLogLikelihoods);
        free(vBitmap);
    }
    return;
}
*/
//Estimates all state vectors from the particles and their weights

void estimateStateVectors(float* aStateVectors, float* aParticles, int _currentLength, int _mNbParticles) {
    //printf("\nEstimating Vectors ... %d%%", 0);
    for (int i = 0; i < _currentLength; i++) {
        //printf("\rEstimating Vectors ... %d%%", ((i+1) * 100)/_currentLength);
        int stateVectorIndex = i * DIM_OF_STATE;
        int stateVectorParticleIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        /*
         * Set the old state to 0
         */
        for (int vI = 0; vI < DIM_OF_STATE; vI++) {
            aStateVectors[stateVectorIndex + vI] = 0.0f;
        }

        for (int pI = 0; pI < _mNbParticles; pI++) {
            int particleIndex = stateVectorParticleIndex + pI * (DIM_OF_STATE + 1);
            for (int vDim = 0; vDim < DIM_OF_STATE; vDim++) {
                aStateVectors[stateVectorIndex + vDim] += aParticles[particleIndex + DIM_OF_STATE] * aParticles[particleIndex + vDim];
            }
        }
    }
    return;
}

/**
 *
 * @param aParticles set of parameters to resample.
 * @return true if resampling was performed, false if not.
 */
bool resample(float* aParticles, int _currentLength, int _mNbParticles) {
    //printf("\nResampling ... %d%%", 0);
    for (int i = 0; i < _currentLength; i++) {
        //printf("\rResampling ... %d%%", ((i+1) * 100)/_currentLength);
        int stateVectorParticleIndex = i * _mNbParticles * (DIM_OF_STATE + 1);
        //
        // First check if the threshold is smaller than Neff
        //
        float vNeff = 0;
        for (int j = 0; j < _mNbParticles; j++) {
            int particleIndex = stateVectorParticleIndex + j * (DIM_OF_STATE + 1);
            vNeff += aParticles[particleIndex + DIM_OF_STATE] * aParticles[particleIndex + DIM_OF_STATE];
        }
        vNeff = 1.0f / vNeff;

        if (vNeff > _mResamplingThreshold) {
            //				System.out.println("no resampling");
            return false; //we won't do the resampling
        }
        //
        // Begin resampling
        //
        //			System.out.println("Resampling");
        float VNBPARTICLES_1 = 1.0f / (float) _mNbParticles;
        double* vC = (double*) malloc(sizeof (double) * (_mNbParticles + 1));
        vC[0] = 0.0;
        for (int vInd = 1; vInd <= _mNbParticles; vInd++) {
            int particleIndex = stateVectorParticleIndex + (vInd - 1) * (DIM_OF_STATE + 1);
            vC[vInd] = vC[vInd - 1] + aParticles[particleIndex + DIM_OF_STATE];
        }

        double vU = (rand() * (double) VNBPARTICLES_1) / RAND_MAX;

        float* vFPParticlesCopy = (float*) malloc(sizeof (float) * _mNbParticles * (DIM_OF_STATE + 1));
        copyStateParticles(vFPParticlesCopy, aParticles, stateVectorParticleIndex, _mNbParticles);
        int vI = 0;
        for (int vParticleCounter = 0; vParticleCounter < _mNbParticles; vParticleCounter++) {
            while (vU > vC[vI] && vI < _mNbParticles) {//this can happen due to numerical reasons
                vI++;
            }
            int sourceParticleIndex = (vI - 1) * (DIM_OF_STATE + 1);
            int destParticleIndex = stateVectorParticleIndex + vParticleCounter * (DIM_OF_STATE + 1);
            //int particleIndex = stateVectorParticleIndex + (vI - 1) * (DIM_OF_STATE + 1);
            for (int vK = 0; vK < DIM_OF_STATE; vK++) {
                aParticles[destParticleIndex + vK] = vFPParticlesCopy[sourceParticleIndex + vK];
            }
            aParticles[destParticleIndex + DIM_OF_STATE] = VNBPARTICLES_1;
            vU += VNBPARTICLES_1;
        }
        free(vC);
        free(vFPParticlesCopy);
    }
    return true;
}

void generateParticlesIntensityBitmap(float* setOfParticles, int stateVectorIndex, bool* vBitmap, int width, int height, int _mNbParticles) {
    // convert to pixel distance and multiply with 3: 4
    float vMaxDistancexy = (3.0f * _mSigmaPSFxy / _spatialRes);
    // get a bounding box around the each feature point
    int vXStart, vXEnd, vYStart, vYEnd;
    //Initiliase array
    for (int vY = 0; vY < height; vY++) {
        int boffset = vY * width;
        for (int vX = 0; vX < width; vX++) {
            vBitmap[boffset + vX] = false;
        }
    }
    for (int i = 0; i < _mNbParticles; i++) {
        int particleIndex = stateVectorIndex + i * (DIM_OF_STATE + 1);
        if (setOfParticles[particleIndex] - vMaxDistancexy < 0) {
            vXStart = 0;
        } else {
            vXStart = (int) boost::math::round<float>(setOfParticles[particleIndex] - vMaxDistancexy);
        }
        if (setOfParticles[particleIndex] + vMaxDistancexy >= width) {
            vXEnd = width - 1;
        } else {
            vXEnd = (int) boost::math::round<float>(setOfParticles[particleIndex] + vMaxDistancexy);
        }
        if (setOfParticles[particleIndex + 1] - vMaxDistancexy < 0) {
            vYStart = 0;
        } else {
            vYStart = (int) boost::math::round<float>(setOfParticles[particleIndex + 1] - vMaxDistancexy);
        }
        if (setOfParticles[particleIndex + 1] + vMaxDistancexy >= height) {
            vYEnd = height - 1;
        } else {
            vYEnd = (int) boost::math::round<float>(setOfParticles[particleIndex + 1] + vMaxDistancexy);
        }
        for (int vY = vYStart; vY <= vYEnd && vY < height; vY++) {
            int boffset = vY * width;
            for (int vX = vXStart; vX <= vXEnd && vX < width; vX++) {
                vBitmap[boffset + vX] = true;
            }
        }
    }
    return;
}

/*void calculateLikelihoods(Matrix aObservationStack, bool* vBitmap, int aFrameIndex, float* vLogLikelihoods, float* mParticles, int stateVectorIndex, int _mNbParticles) {
    //calculate ideal image
    for (int vI = 0; vI < _mNbParticles; vI++) {
        int particleIndex = stateVectorIndex + vI * (DIM_OF_STATE + 1);
        float* vIdealImage = (float*) malloc(sizeof (float) * aObservationStack.width * aObservationStack.height);
        generateIdealImage(mParticles, particleIndex, vIdealImage, aObservationStack.width, aObservationStack.width);
        //calculate likelihood
        vLogLikelihoods[vI] = calculateLogLikelihood(aObservationStack, aFrameIndex, vIdealImage, vBitmap);
        free(vIdealImage);
    }
    return;
}
*/
void generateIdealImage(float* particles, int offset, float* vIdealImage, int width, int height) {
    addBackgroundToImage(vIdealImage, BACKGROUND, width, height);
    addFeaturePointToImage(vIdealImage, particles[offset], particles[offset + 1], particles[offset + 6], width, height);
    return;
}

void addFeaturePointToImage(float* aImage, float x, float y, float aIntensity, int width, int height) {
    float vVarianceXYinPx = _mSigmaPSFxy * _mSigmaPSFxy / (_spatialRes * _spatialRes);
    float vMaxDistancexy = (5.0f * _mSigmaPSFxy / _spatialRes);

    int vXStart, vXEnd, vYStart, vYEnd; //defines a bounding box around the tip
    if (x + .5f - (vMaxDistancexy + .5f) < 0) {
        vXStart = 0;
    } else {
        vXStart = (int) (x + .5f) - (int) (vMaxDistancexy + .5f);
    }
    if (y + .5f - (vMaxDistancexy + .5f) < 0) {
        vYStart = 0;
    } else {
        vYStart = (int) (y + .5f) - (int) (vMaxDistancexy + .5f);
    }
    if (x + .5f + (vMaxDistancexy + .5f) >= width) {
        vXEnd = width - 1;
    } else {
        vXEnd = (int) (x + .5f) + (int) (vMaxDistancexy + .5f);
    }
    if (y + .5f + (vMaxDistancexy + .5f) >= height) {
        vYEnd = height - 1;
    } else {
        vYEnd = (int) (y + .5f) + (int) (vMaxDistancexy + .5f);
    }
    for (int vY = vYStart; vY <= vYEnd && vY < height; vY++) {
        int woffset = vY * width;
        for (int vX = vXStart; vX <= vXEnd && vX < width; vX++) {
            aImage[woffset + vX] += (aIntensity * expf(-(powf(vX - x + .5f, 2.0f) + powf(vY - y + .5f, 2.0f)) / (2.0f * vVarianceXYinPx)));
        }
    }
}

void addBackgroundToImage(float* aImage, float aBackground, int width, int height) {
    for (int y = 0; y < height; y++) {
        int offset = y * width;
        for (int x = 0; x < width; x++) {
            aImage[offset + x] = aBackground;
        }
    }
    return;
}

/**
 * Calculates the likelihood by multipling the poissons marginals around a
 * particle given a image(optimal image)
 *
 * @param aImagePlus: The observed image
 * @param aFrame: The frame index 1<=n<=NSlices(to read out the correct
 * substack from aStack
 * @param aGivenImage: a intensity array, the 'measurement'
 * @param aBitmap: Pixels which are not set to true in the bitmap are pulled
 * out(proportionality)
 * @return the likelihood for the image given
 */
float calculateLogLikelihood(Matrix aStackProcs, int aFrame, float* aGivenImage, bool* aBitmap) //ImageStack aImageStack){
{
    float vLogLikelihood = 0;
    //we need all processors anyway. Profiling showed that the method getProcessor needs a lot of time. Store them
    //in an Array.
    int foffset = aFrame * aStackProcs.width * aStackProcs.height;
    //		long vTime1 = System.currentTimeMillis();
    for (int vY = 0; vY < aStackProcs.height; vY++) {
        int woffset = vY * aStackProcs.width;
        for (int vX = 0; vX < aStackProcs.width; vX++) {
            if (aBitmap[woffset + vX]) {
                float a = -aGivenImage[woffset + vX];
                float b = aStackProcs.elements[foffset + woffset + vX];
                float c = log(aGivenImage[woffset + vX]);
                vLogLikelihood += a + b * c;
            }
        }
    }
    //		System.out.println("used time for loglik = " + (System.currentTimeMillis() - vTime1));
    //IJ.showStatus("likelihood finshed");
    return vLogLikelihood;
}

void runParticleFilter(Matrix aOriginalImage, float* _mParticles, float* _mParticlesMemory, float* _mStateVectors, float* _mStateVectorsMemory, int* _counts, int _currentLength, int _mNbParticles) {
    printf("\nRunning Particle Filter ... %d%%", 0);
    Matrix frame;
    frame.width = aOriginalImage.width;
    frame.height = aOriginalImage.height;
    frame.stride = frame.width;
    frame.depth = 1;
    frame.size = frame.width * frame.height;
    frame.elements = (float*) malloc(sizeof (float) * frame.size);

    for (int vFrameIndex = 0; vFrameIndex < aOriginalImage.depth; vFrameIndex++) {
        printf("\rRunning Particle Filter ... %d%%", (vFrameIndex + 1) * 100 / aOriginalImage.depth);
        matrixCopy(aOriginalImage, frame, vFrameIndex * frame.size);
        // save a copy of the original sigma to restore it afterwards
        float vSigmaOfRWSave[DIM_OF_STATE];
        for (int vI = 0; vI < DIM_OF_STATE; vI++) {
            vSigmaOfRWSave[vI] = _mSigmaOfRandomWalk[vI];
        }
        for (int vRepStep = 0; vRepStep < _mRepSteps; vRepStep++) {
            if (vRepStep == 0) {
                drawNewParticles(_mParticles, _spatialRes, _currentLength, _mNbParticles); //draw the particles at the appropriate position.
                copyStateParticlesToMemory(_mParticlesMemory, _mParticles, vFrameIndex, _mNbParticles, _currentLength);
            } else {
                scaleSigmaOfRW(1.0f / powf(3.0f, (float) vRepStep)); //(1f - (float)vRepStep / (float)mRepSteps);
                DrawParticlesWithRW(_mParticles, _currentLength, _mNbParticles);
            }
            updateParticleWeightsOnGPU(frame, _mParticles, _currentLength, _mNbParticles);
            //updateParticleWeights(aOriginalImage, vFrameIndex);
            estimateStateVectors(_mStateVectors, _mParticles, _currentLength, _mNbParticles);

            int killed = checkStateVectors(_mStateVectors, _mParticles, frame.width, frame.height, _currentLength, _mNbParticles);
            _currentLength -= killed;
            if (!resample(_mParticles, _currentLength, _mNbParticles)) {//further iterations are not necessary.
                //						System.out.println("number of iterations needed at this frame: " + vRepStep);
                break;
            }
        }
        //restore the sigma vector
        for (int vI = 0; vI < DIM_OF_STATE; vI++) {
            _mSigmaOfRandomWalk[vI] = vSigmaOfRWSave[vI];
        }
        //save the new states
        copyStateVector(_mStateVectorsMemory, _mStateVectors, vFrameIndex, _currentLength);
        _counts[vFrameIndex] = _currentLength;
    }
    free(frame.elements);
}

/**
 * Draws new particles for all the objects
 *
 */
void drawNewParticles(float* aParticlesToRedraw, float spatialRes, int _currentLength, int _mNbParticles) {
    //invoke this method here to not repeat it for every particle, pass it by argument
    //TODO: Better would probably be to scale the sigma vector from beginning; this would then introduce the problem that 
    //		the sampling of the particles cannot be treated for each dimension independently, which introduces
    //		an error.
    for (int i = 0; i < _currentLength; i++) {
        int stateVectorParticleIndex = i * (DIM_OF_STATE + 1) * _mNbParticles;
        for (int j = 0; j < _mNbParticles; j++) {
            int particleIndex = stateVectorParticleIndex + j * (DIM_OF_STATE + 1);
            drawFromProposalDistribution(aParticlesToRedraw, spatialRes, particleIndex);
        }
    }
}

void drawFromProposalDistribution(float* particles, float spatialRes, int particleIndex) {
    particles[3 + particleIndex] = particles[3 + particleIndex] + var_nor() * (_mSigmaOfDynamics[0] / spatialRes);
    particles[4 + particleIndex] = particles[4 + particleIndex] + var_nor() * (_mSigmaOfDynamics[1] / spatialRes);
    particles[5 + particleIndex] = particles[5 + particleIndex] + var_nor() * (_mSigmaOfDynamics[2] / spatialRes);
    particles[particleIndex] = particles[particleIndex] + particles[3 + particleIndex];
    particles[1 + particleIndex] = particles[1 + particleIndex] + particles[4 + particleIndex];
    particles[2 + particleIndex] = particles[2 + particleIndex] + particles[5 + particleIndex];
    particles[6 + particleIndex] = particles[6 + particleIndex] + var_nor() * _mSigmaOfDynamics[3];
    if (particles[6 + particleIndex] < BACKGROUND + 1) {
        particles[6 + particleIndex] = BACKGROUND + 1;
    }
}

/*
 * Check to ensure that all stateVectors represent objects within the field of view. Any that are not are "killed".
 */
int checkStateVectors(float* stateVectors, float* particles, int width, int height, int nVectors, int nParticles) {
    int killed = 0;
    int vEnd = nVectors * DIM_OF_STATE;
    int pEnd = nVectors * nParticles * (DIM_OF_STATE + 1);
    for (int v = 0, i = 0; v < nVectors; v++) {
        int svIndex = i * (DIM_OF_STATE);
        float x = stateVectors[svIndex];
        float y = stateVectors[svIndex + 1];
        // If (x, y) is outside image boundary, shift all subsequent state vectors forward one slot
        if (x < 0.0f || x >= width || y < 0.0f || y >= height) {
            int pOffset = nParticles * (DIM_OF_STATE + 1);
            int pIndex = i * pOffset;
            for (int j = svIndex + DIM_OF_STATE; j < vEnd; j++) {
                stateVectors[j - DIM_OF_STATE] = stateVectors[j];
            }
            for (int k = pIndex + pOffset; k < pEnd; k++) {
                particles[k - pOffset] = particles[k];
            }
            killed++;
        } else {
            i++;
        }
    }
    return killed;
}

void output(int* dims, int frames, string outputDir, int _mNbParticles, int* _counts, float* _mParticlesMemory) {
    printf("\nBuilding output ... %d%%", 0);
    Matrix output;
    output.width = dims[0] * _scalefactor;
    output.stride = output.width;
    output.height = dims[1] * _scalefactor;
    output.size = output.width * output.height;
    output.elements = (float*) malloc(sizeof (float) * output.size);

    for (int frameIndex = 0; frameIndex < frames; frameIndex++) {
        printf("\rBuilding output ... %d%%", (frameIndex + 1) * 100 / frames);
        int stateVectorFrameIndex = frameIndex * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1);
        for (int p = 0; p < output.width * output.height; p++) {
            output.elements[p] = 0.0f;
        }
        Mat cudasaveframe(dims[1] * _scalefactor, dims[0] * _scalefactor, CV_32F);
        for (int i = 0; i < _counts[frameIndex]; i++) {
            int stateVectorIndex = stateVectorFrameIndex + i * _mNbParticles * (DIM_OF_STATE + 1);
            /*addFeaturePointToImage(output.elements, _mStateVectorsMemory[stateVectorIndex] * _scalefactor,
                    _mStateVectorsMemory[stateVectorIndex + 1] * _scalefactor,
                    _mStateVectorsMemory[stateVectorIndex + 6], output.width, output.height);*/
            for (int j = 0; j < _mNbParticles; j++) {
                int particleIndex = stateVectorIndex + j * (DIM_OF_STATE + 1);
                int x = (int) boost::math::round<float>(_mParticlesMemory[particleIndex] * _scalefactor);
                int y = (int) boost::math::round<float>(_mParticlesMemory[particleIndex + 1] * _scalefactor);
                //addFeaturePointToImage(output.elements, x, y, _mParticles[particleIndex + DIM_OF_STATE], output.width, output.height);
                if (x > 0 && x < output.width && y > 0 && y < output.height) {
                    int index = x + y * output.stride;
                    output.elements[index] += 1.0f;
                }
            }
        }
        copyFromMatrix(cudasaveframe, output, 0, 1.0f);
        /*for(int j = 0; j < frameIndex; j++){
                int stateVectorFrameIndex = j * MAX_DETECTIONS * DIM_OF_STATE;
                for(int k = 0; k < _counts[j]; k++){
                        int stateVectorIndex = stateVectorFrameIndex + k * DIM_OF_STATE;
                        Point centre((int)round(_mStateVectorsMemory[stateVectorIndex] * _scalefactor), (int)round(_mStateVectorsMemory[stateVectorIndex + 1] * _scalefactor));
                        circle(cudasaveframe, centre, FIT_RADIUS, 255);
                }
        }*/
        cudasaveframe.convertTo(cudasaveframe, CV_16UC1);
        string savefilename(outputDir);
        savefilename.append("/");
        savefilename.append(boost::lexical_cast<string > (frameIndex));
        savefilename.append(".tif");
        imwrite(savefilename, cudasaveframe);
    }
    return;
}

//Copy statevectors corresponding to frame index from source to dest

void copyStateVector(float* dest, float* source, int index, int _currentLength) {
    int boffset = index * MAX_DETECTIONS * DIM_OF_STATE;
    for (int i = 0; i < _currentLength; i++) {
        int soffset = i * DIM_OF_STATE;
        for (int j = 0; j < DIM_OF_STATE; j++) {
            dest[boffset + soffset + j] = source[soffset + j];
        }
    }
    return;
}

void copyStateParticles(float* dest, float* source, int stateVectorParticleIndex, int _mNbParticles) {
    for (int i = 0; i < _mNbParticles; i++) {
        int particleIndex = stateVectorParticleIndex + i * (DIM_OF_STATE + 1);
        for (int j = 0; j < DIM_OF_STATE + 1; j++) {
            dest[j + particleIndex - stateVectorParticleIndex] = source[j + particleIndex];
        }
    }
    return;
}

void copyStateParticlesToMemory(float* dest, float* source, int frameIndex, int _mNbParticles, int _currentLength) {
    int frameVectorIndex = frameIndex * MAX_DETECTIONS * _mNbParticles * (DIM_OF_STATE + 1);
    for (int k = 0; k < _currentLength; k++) {
        int stateVectorIndex = frameVectorIndex + _mNbParticles * (DIM_OF_STATE + 1) * k;
        for (int i = 0; i < _mNbParticles; i++) {
            int particleIndex = stateVectorIndex + i * (DIM_OF_STATE + 1);
            for (int j = 0; j < DIM_OF_STATE + 1; j++) {
                dest[j + particleIndex] = source[j + particleIndex - frameVectorIndex];
            }
        }
    }
    return;
}

void copyParticle(float* dest, float* source, int index) {
    for (int i = 0; i < DIM_OF_STATE + 1; i++) {
        dest[i + index] = source[i];
    }
    return;
}

void updateStateVector(float* vector, int index, float* _mStateVectors) {
    int offset = index * DIM_OF_STATE;
    for (int i = 0; i < DIM_OF_STATE; i++) {
        _mStateVectors[i + offset] = vector[i];
    }
    return;
}
