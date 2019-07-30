# distutils: language = c++
import cython
import numpy as np 
cimport numpy as np 
from libcpp.vector cimport vector
from libcpp.pair cimport pair

np.import_array()

# Declare the interface to the C code
cdef extern from "dummy.cpp":
    cdef pair[vector[int], vector[vector[double]]] c_compute (double* array, double multiplier, int m, int n)

@cython.boundscheck(False)
@cython.wraparound(False)

def compute(np.ndarray[double, ndim = 1, mode = "c"] matrix, double value):
    cdef int m, n 
    m, n = matrix.shape[0] / 2, 2
    res = c_compute(&matrix[0], value, m, n)

    return res 
    