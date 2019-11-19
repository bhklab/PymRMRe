#ifndef mRMRe_exports_h
#define mRMRe_exports_h

#include <cstdlib>
#include <vector>
#include <tuple>
#include <utility>

#include "Filter.h"
#include "Math.h"
#include "MutualInformationMatrix.h"

using namespace std;

// Need to add the size of level vector (actually just the feature count per solution)

std::pair <vector<vector<int> >, vector<vector<vector<double> > > >
c_export_filters(const int * const childrenCountPerLevel, 
                const unsigned int levelCount,
                double* const dataMatrix, 
                double* const priorsMatrix, 
                const unsigned int priorsCount,
                const double priorsWeight,
                const int* const sampleStrata, 
                const double* const sampleWeights, 
                const int* const featureTypes, 
                const unsigned int sampleCount, 
                const unsigned int featureCount, 
                const unsigned int sampleStratumCount, 
                unsigned int* targetFeatureIndices, 
                const unsigned int fixedFeatureCount,
                const unsigned int targetCount,
                const unsigned int continuousEstimator, 
                const unsigned int outX, 
                const unsigned int bootstrapCount, 
                double* const miMatrix);
        // priorsMatrix is array? Instead of matrix? 
        // dataMatrix is array? Instead of matrix?
        // FeatureCount is array? Instead of integer? 
        // priorsWeight are sured to be the scalar
        // miMatrix is array?
        ////// The data types (const stuff) have been all clear, except the targetFeatureIndices 

/* 
std::pair <vector<vector<int> >, vector<vector<vector<double> > > >
c_export_filters_bootstrap(unsigned int const solutionCount, unsigned int const solutionLength, double* const dataMatrix, double* const priorsMatrix,
        double const priorsWeight, int const* const sampleStrata, double const* const sampleWeights, int const* const featureTypes, 
        unsigned int const sampleCount, unsigned int const featureCount, unsigned int const sampleStratumCount, unsigned int* targetFeatureIndices, 
        unsigned int const continuousEstimator, unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix);

        ////// The data types (const stuff) have been all clear, except the targetFeatureIndices 

*/

void
c_export_mim(double* const dataMatrix, 
           double* const priorsMatrix, 
           const double priorsWeight, 
           const int* const sampleStrata, 
           const double* const sampleWeights,
           const int* const featureTypes, 
           const unsigned int sampleCount, 
           const unsigned int featureCount, 
           const unsigned int sampleStratumCount, 
           const unsigned int continuousEstimator, 
           const unsigned int outX, 
           const unsigned int bootstrapCount, 
           double* const miMatrix);

void 
get_thread_count(unsigned int threadCount);

void 
set_thread_count(unsigned int const threadCount);


#endif /* mRMRe_exports_h */
