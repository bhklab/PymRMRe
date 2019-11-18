# cexpt.pxd
# 
# Declarations of "external" C++ functions and structures
cimport cython
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "exports.h":
    pair[vector[int[:]], vector[vector[double[:]]]] export_filters(int[:] childrenCountPerLevel, 
        double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight,
        int[:] sampleStrata, double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, 
        unsigned int featureCount, unsigned int sampleStratumCount, unsigned int* targetFeatureIndices, unsigned int continuousEstimator, 
        unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)

    pair[vector[int[:]], vector[vector[double[:]]]] export_filters_bootstrap(unsigned int solutionCount, 
        unsigned int solutionLength, double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, 
        unsigned int sampleStratumCount, unsigned int[:] targetFeatureIndices, unsigned int continuousEstimator, 
        unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)
    
    void export_mim(double[:] dataMatrix, double[:] priorsMatrix, double priorsWeight, int[:] sampleStrata, 
        double[:] sampleWeights, int[:] featureTypes, unsigned int sampleCount, unsigned int featureCount, unsigned int sampleStratumCount, 
        unsigned int continuousEstimator, unsigned int outX, unsigned int bootstrapCount, double[:] miMatrix)

