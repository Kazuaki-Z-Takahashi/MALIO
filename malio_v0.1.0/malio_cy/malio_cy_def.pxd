from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc cimport math
from libcpp.vector cimport vector
cimport openmp
ctypedef vector[double] dvec
ctypedef vector[dvec] dvec_vec
ctypedef vector[dvec_vec] dvec_vec_vec
ctypedef vector[int] ivec
ctypedef vector[ivec] ivec_vec
ctypedef vector[double complex] cvec
ctypedef vector[cvec] cvec_vec
