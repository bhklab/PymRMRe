#include "exports.h"
#include <iostream>
#include <array>
#include <iterator>

using namespace std;

//Exports::Exports(int )
// Constructor
Exports::Exports(int** _solutions, double** _causality, double** _scores) {
    this->solutions = _solutions;
    this->causality = _causality;
    this->scores = _scores;
}

// Destructor
Exports::~Exports() {
    for (int i = 0; i < this->solutions.size(); i++) {
        delete[] this->solutions[i]; 
        delete[] this->scores[i];
        delete[] this->causality[i];
    }
    delete[] this->solutions;
    delete[] this->scores;
    delete[] this->causality;
}

Exports
Exports::export_filters(int const* const childrenCountPerLevel, double* const dataMatrix, double* const priorsMatrix, double const priorsWeight,
            int const* const sampleStrata, double const* const sampleWeights, int const* const featureTypes, unsigned int const sampleCount, 
            unsigned int const featureCount, unsigned int const sampleStratumCount, unsigned int* targetFeatureIndices, unsigned int const continuousEstimator, 
            unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix)
{
    Matrix const priors_matrix(priorsMatrix, featureCount, featureCount);

    Matrix const* const p_priors_matrix = 
            std::size(priorsMatrix) == featureCount * featureCount ? &priors_matrix : 0;

    Data data(dataMatrix, p_priors_matrix, priorsWeight, sampleCount, featureCount, sampleStrata, sampleWeights,
            featureTypes, sampleStratumCount, continuousEstimator, outX != 0, bootstrapCount);

    MutualInformationMatrix mi_matrix(&data, miMatrix);

    unsigned int solution_count = 1;
    for (unsigned int i = 0; i < std::size(childrenCountPerLevel); ++i)
        solution_count *= childrenCountPerLevel[i];
    
    unsigned int const feature_count_per_solution = std::size(childrenCountPerLevel);
    unsigned int const chunk_size = solution_count * feature_count_per_solution;
    
    // Return value
    Exports expt;
    // SET_VECTOR_ELT about the result
    int targetFeatureLength = std::size(targetFeatureIndices);
    
    expt.solutions = new int*[targetFeatureLength];
    expt.causality = new double*[targetFeatureLength];
    expt.scores = new double*[targetFeatureLength];

    for (unsigned int i = 0; i < targetFeatureLength; ++i)
    {
        Filter filter(childrenCountPerLevel, std::size(childrenCountPerLevel), &mi_matrix, targetFeatureIndices[i]);
        filter.build();
        
        expt.solutions[i] = new int[chunk_size];
        expt.causality[i] = new double[featureCount];
        expt.scores[i] = new double[chunk_size];

        filter.getSolutions(expt.solutions[i]);
        filter.getScores(expt.scores[i]);

        for (unsigned int k = 0; k < featureCount; ++k) 
            expt.causality[i][k] = std::numeric_limits<double>::quiet_NaN();
        
        Math::computeCausality(expt.causality[i], &mi_matrix, expt.solutions[i], solution_count,
                feature_count_per_solution, featureCount, targetFeatureIndices[i]);
    }

    return expt;
}

Exports
Exports::export_filters_bootstrap(unsigned int const solutionCount, unsigned int const solutionLength, double* const dataMatrix, double* const priorsMatrix,
            double const priorsWeight, int const* const sampleStrata, double const* const sampleWeights, int const* const featureTypes, 
            unsigned int const sampleCount, unsigned int const featureCount, unsigned int const sampleStratumCount, unsigned int* targetFeatureIndices, 
            unsigned int const continuousEstimator, unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix)
{
    Matrix const priors_matrix(priorsMatrix, featureCount, featureCount);
    Matrix const* const p_priors_matrix = 
            std::size(priorsMatrix) == featureCount * featureCount ? &priors_matrix : 0;
    Data data(dataMatrix, p_priors_matrix, priorsWeight, sampleCount, featureCount, sampleStrata, sampleWeights,
            featureTypes, sampleStratumCount, continuousEstimator, outX != 0, bootstrapCount);

    // No need for the mutual information matrix
    unsigned int solution_count = solutionCount;
    unsigned int const feature_count_per_solution = solutionLength;
    unsigned int const chunk_size = solution_count * feature_count_per_solution;

    int * const p_children_count_per_level = new int[feature_count_per_solution];
    for (unsigned int i = 0; i < feature_count_per_solution; ++i)
        p_children_count_per_level[i] = 1;
    
    // Return value
    Exports expt;
    // SET_VECTOR_ELT about the result
    int targetFeatureLength = std::size(targetFeatureIndices);
    expt.solutions = new int*[targetFeatureLength];
    expt.causality = new double*[targetFeatureLength];
    expt.scores = new double*[targetFeatureLength];

    for (unsigned int i = 0; i < targetFeatureLength; ++i)
    {
        
        expt.solutions[i] = new int[chunk_size];
        expt.causality[i] = new double[featureCount];
        expt.scores[i] = new double[chunk_size];

        for (unsigned int k = 0; k < featureCount; ++k) 
            expt.causality[i][k] = std::numeric_limits<double>::quiet_NaN();

    }

    for (unsigned int i = 0; i < solution_count; ++i)
    {
        MutualInformationMatrix mi_matrix(&data);

        for (unsigned int j = 0; j < targetFeatureLength; ++j) 
        {
            Filter filter(p_children_count_per_level, feature_count_per_solution, &mi_matrix, targetFeatureIndices[j]);
            filter.build();
            filter.getSolutions(expt.solutions[j] + i * feature_count_per_solution);
        }
        data.bootstrap();
    }

    delete[] p_children_count_per_level;
    return expt;

}

void
Exports::export_mim(double* const dataMatrix, double* const priorsMatrix, double const priorsWeight, int const* const sampleStrata, double const* const sampleWeights,
            int const* const featureTypes, unsigned int const sampleCount, unsigned int const featureCount, unsigned int const sampleStratumCount, 
            unsigned int const continuousEstimator, unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix)
{
    Matrix const priors_matrix(priorsMatrix, featureCount, featureCount);
    Matrix const* const p_priors_matrix = 
            std::size(priorsMatrix) == featureCount * featureCount ? &priors_matrix : 0;
    Data data(dataMatrix, p_priors_matrix, priorsWeight, sampleCount, featureCount, sampleStrata, sampleWeights,
            featureTypes, sampleStratumCount, continuousEstimator, outX != 0, bootstrapCount);
    MutualInformationMatrix mi_matrix(&data, miMatrix);
    mi_matrix.build();
    return;
}

void 
Exports::get_thread_count(unsigned int threadCount)
{
#ifdef _OPENMP
    threadCount = omp_get_max_threads();
#endif

    return;
}

void 
Exports::set_thread_count(unsigned int const threadCount)
{
#ifdef _OPENMP
    opm_set_num_threads(threadCount);
#endif

    return;
}