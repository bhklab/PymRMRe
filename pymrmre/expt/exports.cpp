#include "exports.h"
#include <iostream>
#include <array>
#include <iterator>
#include <tuple>
#include <utility>

using namespace std;

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
                double* const miMatrix)
{
    Matrix const priors_matrix(priorsMatrix, featureCount, featureCount);
    
    Matrix const* const p_priors_matrix = 
            priorsCount == featureCount * featureCount ? &priors_matrix : 0;
    
    Data data(dataMatrix, p_priors_matrix, priorsWeight, sampleCount, featureCount, sampleStrata, sampleWeights,
            featureTypes, sampleStratumCount, continuousEstimator, outX != 0, bootstrapCount);
    
    MutualInformationMatrix mi_matrix(&data, miMatrix);

    unsigned int solution_count = 1;
    for (unsigned int i = 0; i < levelCount; ++i)
        solution_count *= childrenCountPerLevel[i];
    
    unsigned int const feature_count_per_solution = levelCount;
    unsigned int const chunk_size = solution_count * feature_count_per_solution;
    
    // Return value
    vector<vector<int> > solutions;
    vector<vector<vector<double> > > expt;
    vector<vector<double> > causality, scores;

    expt.push_back(causality);
    expt.push_back(scores);

    unsigned int targetFeatureLength = targetCount;


    for (unsigned int i = 0; i < targetFeatureLength; ++i)
    {   
        // Build new vector
        vector<int> solutions_i;
        vector<double> casuality_i, scores_i;

        Filter filter(childrenCountPerLevel, levelCount, &mi_matrix, targetFeatureIndices[i], fixedFeatureCount);
        filter.build();
        
        int* sol = new int[chunk_size];
        double* cas = new double[featureCount];
        double* sc = new double[chunk_size];

        filter.getSolutions(sol);
        filter.getScores(sc);

        for (unsigned int k = 0; k < featureCount; ++k) 
            cas[k] = std::numeric_limits<double>::quiet_NaN();
        
        Math::computeCausality(cas, &mi_matrix, sol, solution_count,
                feature_count_per_solution, featureCount, targetFeatureIndices[i]);

        for (unsigned int k = 0; k < chunk_size; ++k) {
            solutions_i.push_back(sol[k]);
            scores_i.push_back(sc[k]);
        }
        for (unsigned int k = 0; k < featureCount; ++k) 
            casuality_i.push_back(cas[i]);

        solutions.push_back(solutions_i);
        expt[0].push_back(casuality_i);
        expt[1].push_back(scores_i);
        
        delete[] sol;
        delete[] cas;
        delete[] sc;
    }

    return std::make_pair(solutions, expt);
}

/* 
std::pair <vector<vector<int>>, vector<vector<vector<double>>>>
c_export_filters_bootstrap(unsigned int const solutionCount, unsigned int const solutionLength, double* const dataMatrix, double* const priorsMatrix,
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
    vector<vector<int>> solutions;
    vector<vector<vector<double>>> expt;
    vector<vector<double>> causality, scores;

    expt.push_back(causality);
    expt.push_back(scores);
    // SET_VECTOR_ELT about the result
    int targetFeatureLength = std::size(targetFeatureIndices);

    for (unsigned int i = 0; i < targetFeatureLength; ++i)
    {
        // Build new vector
        //vector<int> solutions_i;
        vector<double> casuality_i;
        //solutions.push_back(solutions_i);
        expt[0].push_back(casuality_i);
        //expt[1].push_back(scores_i);

        //int* sol = new int[chunk_size];
        double* cas = new double[featureCount];
        //double* sc = new double[chunk_size];

        for (unsigned int k = 0; k < featureCount; ++k) {
            cas[k] = std::numeric_limits<double>::quiet_NaN();
            expt[0][i].push_back(cas[k]);
        }
    }

    for (unsigned int i = 0; i < solution_count; ++i)
    {
        MutualInformationMatrix mi_matrix(&data);

        for (unsigned int j = 0; j < targetFeatureLength; ++j) 
        {
            vector<int> solutions_i;
            vector<double> scores_i;

            int* sol = new int[chunk_size];
            double* sc = new double[chunk_size];

            Filter filter(p_children_count_per_level, feature_count_per_solution, &mi_matrix, targetFeatureIndices[j]);
            filter.build();
            //filter.getSolutions(solutions[j] + i * feature_count_per_solution);
        }
        data.bootstrap();
    }

    delete[] p_children_count_per_level;
    return std::make_pair(solutions, expt);

} */

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
           double* const miMatrix)
{
    Matrix const priors_matrix(priorsMatrix, featureCount, featureCount);
    Matrix const* const p_priors_matrix = 
            (40000) == featureCount * featureCount ? &priors_matrix : 0;
    Data data(dataMatrix, p_priors_matrix, priorsWeight, sampleCount, featureCount, sampleStrata, sampleWeights,
            featureTypes, sampleStratumCount, continuousEstimator, outX != 0, bootstrapCount);
    MutualInformationMatrix mi_matrix(&data, miMatrix);
    mi_matrix.build();
    return;
}

// void 
// get_thread_count(unsigned int threadCount)
// {
// #ifdef _OPENMP
//     threadCount = omp_get_max_threads();
// #endif

//     return;
// }

// void 
// set_thread_count(unsigned int const threadCount)
// {
// #ifdef _OPENMP
//     opm_set_num_threads(threadCount);
// #endif

//     return;
// }
