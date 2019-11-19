# distutils: language = c++
import cython
import numpy as np 
cimport numpy as np 
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libc.stdint cimport int32_t, int64_t

np.import_array()

# Create a Cython extension type which holds a C++ instance as an attribute and create a 
# bunch of forwarding methods Python extension type.
__version__ = '0.1.0'

cdef extern from "exports.cpp":
    # cdef fused int_type:
    #    np.int32_t
        #np.int64_t
    
    cdef pair[vector[vector[int]], vector[vector[vector[double]]]]  c_export_filters(const int* const childrenCountPerLevel, 
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

    cdef void c_export_mim(double* const dataMatrix, 
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
    
#    cdef pair[vector[vector[int]], vector[vector[vector[double]]]] export_filters_bootstrap(unsigned int solutionCount, 
#        unsigned int solutionLength, double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
#        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, 
#        unsigned int sampleStratumCount, unsigned int[:] targetFeatureIndices, unsigned int continuousEstimator, 
#        unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix) except + 
    
#    cdef void export_mim(double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
#        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, unsigned int sampleStratumCount, 
#        unsigned int continuousEstimator, unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)

@cython.boundscheck(False)
@cython.wraparound(False)

def export_filters(np.ndarray[int, ndim = 1, mode = "c"] childrenCountPerLevel, 
                    int levelCount,
                    np.ndarray[double, ndim = 1, mode = "c"] dataMatrix, 
                    np.ndarray[double, ndim = 1, mode = "c"] priorsMatrix, 
                    int priorsCount,
                    double priorsWeight, 
                    np.ndarray[int, ndim = 1, mode = "c"] sampleStrata, 
                    np.ndarray[double, ndim = 1, mode = "c"] sampleWeights,
                    np.ndarray[int, ndim = 1, mode = "c"] featureTypes, 
                    int sampleCount, 
                    int featureCount, 
                    int sampleStratumCount, 
                    np.ndarray[unsigned int, ndim = 1, mode = "c"] targetFeatureIndices, 
                    int fixedFeatureCount,
                    int targetCount,
                    int continuousEstimator,
                    int outX, 
                    int bootstrapCount, 
                    np.ndarray[double, ndim = 1, mode = "c"] miMatrix):
    
    res = c_export_filters(&childrenCountPerLevel[0], levelCount, &dataMatrix[0], &priorsMatrix[0], priorsCount,
                            priorsWeight, &sampleStrata[0], &sampleWeights[0], &featureTypes[0], sampleCount, 
                            featureCount, sampleStratumCount, &targetFeatureIndices[0], fixedFeatureCount, 
                            targetCount, continuousEstimator, outX, bootstrapCount, &miMatrix[0])

    return res

def export_mim(np.ndarray[double, ndim = 1, mode = "c"] dataMatrix, 
                np.ndarray[double, ndim = 1, mode = "c"] priorsMatrix,
                double priorsWeight,
                np.ndarray[int, ndim = 1, mode = "c"] sampleStrata,
                np.ndarray[double, ndim = 1, mode = "c"] sampleWeights,
                np.ndarray[int, ndim = 1, mode = "c"] featureTypes,
                int sampleCount,
                int featureCount,
                int sampleStratumCount,
                int continuousEstimator,
                int outX,
                int bootstrapCount,
                np.ndarray[double, ndim = 1, mode = "c"] miMatrix):
    
    c_export_mim(&dataMatrix[0], &priorsMatrix[0], priorsWeight, &sampleStrata[0], &sampleWeights[0],
                 &featureTypes[0], sampleCount, featureCount, sampleStratumCount, continuousEstimator,
                 outX, bootstrapCount, &miMatrix[0])
    return 
