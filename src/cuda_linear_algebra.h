#ifndef CUDA_LA_H
#define CUDA_LA_H

#include "linear_algebra.h"
#include <cuda_runtime.h>

__global__ void cuda_vector_add(double *a, double *b, double *result,
                                uint32_t size);
__global__ void cuda_vector_scale(double *v, double scalar, double *result,
                                  uint32_t size);
__global__ void cuda_matrix_add(double *a, double *b, double *result,
                                uint32_t rows, uint32_t cols);
__global__ void cuda_matrix_multiply(double *a, double *b, double *result,
                                     uint32_t m, uint32_t n, uint32_t k);

Vector *cuda_vector_add_wrapper(Vector *a, Vector *b);
Vector *cuda_vector_scale_wrapper(Vector *v, double scalar);
Matrix *cuda_matrix_add_wrapper(Matrix *a, Matrix *b);
Matrix *cuda_matrix_multiply_wrapper(Matrix *a, Matrix *b);

// Helper function declarations
double *matrix_to_array(Matrix *m);
void array_to_matrix(double *arr, Matrix *m);

#endif
