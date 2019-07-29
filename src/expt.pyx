# distutils: language = c++
from libcpp.vector cimport vector
from libcpp.pair cimport pair

#np.import_array()

# Create a Cython extension type which holds a C++ instance as an attribute and create a 
# bunch of forwarding methods Python extension type.
__version__ = '1.0.0'

cdef extern from "exports.h":
    pair[vector[vector[int]], vector[vector[vector[double]]]] export_filters(int[:] childrenCountPerLevel, 
        double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight,
        int[:] sampleStrata, double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, 
        unsigned int featureCount, unsigned int sampleStratumCount, unsigned int* targetFeatureIndices, unsigned int continuousEstimator, 
        unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)
    
    pair[vector[vector[int]], vector[vector[vector[double]]]] export_filters_bootstrap(unsigned int solutionCount, 
        unsigned int solutionLength, double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, 
        unsigned int sampleStratumCount, unsigned int[:] targetFeatureIndices, unsigned int continuousEstimator, 
        unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)
    
    void export_mim(double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, unsigned int sampleStratumCount, 
        unsigned int continuousEstimator, unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)
    

    