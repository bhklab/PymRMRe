#ifndef mRMRe_exports_h
#define mRMRe_exports_h

#include <cstdlib>
#include <vector>

#include "Filter.h"
#include "Math.h"
#include "MutualInformationMatrix.h"

using namespace std;

/* 
struct Result
{
    int** solutions;
    double** causality;
    double** scores;
};*/

class Exports
{
public:
    //void
    //export_concordance_index();
    int** solutions;
    double** causality;
    double** scores;
    Exports();
    Exports(int** _solutions, double** _causality, double** _scores);
    /* 
    Exports(int** _solutions, double** _causality, double** _scores) {
        solutions = _solutions;
        causality = _causality;
        scores = _scores;
    }
    */
    ~Exports();

    Exports 
    export_filters(int const* const childrenCountPerLevel, double* const dataMatrix, double* const priorsMatrix, double const priorsWeight,
            int const* const sampleStrata, double const* const sampleWeights, int const* const featureTypes, unsigned int const sampleCount, 
            unsigned int const featureCount, unsigned int const sampleStratumCount, unsigned int* targetFeatureIndices, unsigned int const continuousEstimator, 
            unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix);
        // priorsMatrix is array? Instead of matrix? 
        // dataMatrix is array? Instead of matrix?
        // FeatureCount is array? Instead of integer? 
        // priorsWeight are sured to be the scalar
        // miMatrix is array?
        ////// The data types (const stuff) have been all clear, except the targetFeatureIndices 
    
    Exports 
    export_filters_bootstrap(unsigned int const solutionCount, unsigned int const solutionLength, double* const dataMatrix, double* const priorsMatrix,
            double const priorsWeight, int const* const sampleStrata, double const* const sampleWeights, int const* const featureTypes, 
            unsigned int const sampleCount, unsigned int const featureCount, unsigned int const sampleStratumCount, unsigned int* targetFeatureIndices, 
            unsigned int const continuousEstimator, unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix);

        ////// The data types (const stuff) have been all clear, except the targetFeatureIndices 

    void
    export_mim(double* const dataMatrix, double* const priorsMatrix, double const priorsWeight, int const* const sampleStrata, double const* const sampleWeights,
            int const* const featureTypes, unsigned int const sampleCount, unsigned int const featureCount, unsigned int const sampleStratumCount, 
            unsigned int const continuousEstimator, unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix);

    void 
    get_thread_count(unsigned int threadCount);

    void 
    set_thread_count(unsigned int const threadCount);

};


#endif /* mRMRe_exports_h */