# distutils: language = c++
import cython
import numpy as np 
cimport numpy as np 
from libcpp.vector cimport vector
from libcpp.pair cimport pair

np.import_array()

# Create a Cython extension type which holds a C++ instance as an attribute and create a 
# bunch of forwarding methods Python extension type.
__version__ = '1.0.0'

cdef extern from "exports.h":
    cdef pair[vector[vector[int]], vector[vector[vector[double]]]] c_export_filters(int const* const childrenCountPerLevel, double* const dataMatrix, 
        double* const priorsMatrix, double const priorsWeight,
        int const* const sampleStrata, double const* const sampleWeights, int const* const featureTypes, unsigned int const sampleCount, 
        unsigned int const featureCount, unsigned int const sampleStratumCount, unsigned int* targetFeatureIndices, unsigned int const continuousEstimator, 
        unsigned int const outX, unsigned int const bootstrapCount, double* const miMatrix)
    
    cdef pair[vector[vector[int]], vector[vector[vector[double]]]] export_filters_bootstrap(unsigned int solutionCount, 
        unsigned int solutionLength, double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, 
        unsigned int sampleStratumCount, unsigned int[:] targetFeatureIndices, unsigned int continuousEstimator, 
        unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix) except + 
    
    cdef void export_mim(double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, unsigned int sampleStratumCount, 
        unsigned int continuousEstimator, unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)



def export_filters(np.ndarray[int, ndim = 1, mode = "c"] childrenCountPerLevel, 
                    np.ndarray[double, ndim = 1, mode = "c"] dataMatrix, 
                    np.ndarray[double, ndim = 1, mode = "c"] priorsMatrix, 
                    np.ndarray[double, ndim = 1, mode = "c"] priorsWeight, 
                    np.ndarray[int, ndim = 1, mode = "c"] sampleStrata, 
                    np.ndarray[double, ndim = 1, mode = "c"] sampleWeights,
                    np.ndarray[int, ndim = 1, mode = "c"] featureTypes, 
                    int sampleCount, 
                    int featureCount, 
                    int sampleStratumCount, 
                    np.ndarray[int, ndim = 1, mode = "c"] targetFeatureIndices, 
                    int continuousEstimator,
                    int outX, 
                    int bootstrapCount, 
                    np.ndarray[double, ndim = 1, mode = "c"] miMatrix):
    
    res = c_export_filters(&childrenCountPerLevel[0], &dataMatrix[0], &priorsMatrix[0], &priorsWeight[0],
                            &sampleStrata[0], &sampleWeights[0], &featureTypes[0], sampleCount, featureCount,
                            sampleStratumCount, &targetFeatureIndices[0], continuousEstimator, outX, 
                            bootstrapCount, &miMatrix[0])

    return res


    