#include "cuda_linear_algebra.h"
#include "linear_algebra.h"
#include <cuda_runtime.h>

__global__ void cuda_vector_add(double *a, double *b, double *result,
                                uint32_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + b[idx];
  }
}

__global__ void cuda_vector_scale(double *v, double scalar, double *result,
                                  uint32_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = v[idx] * scalar;
  }
}

__global__ void cuda_matrix_add(double *a, double *b, double *result,
                                uint32_t rows, uint32_t cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows && col < cols) {
    int idx = row * cols + col;
    result[idx] = a[idx] + b[idx];
  }
}

double *matrix_to_array(Matrix *m) {
  double *arr = (double *)malloc(m->rows * m->cols * sizeof(double));
  for (uint32_t i = 0; i < m->rows; i++) {
    for (uint32_t j = 0; j < m->cols; j++) {
      arr[i * m->cols + j] = m->data[i][j];
    }
  }
  return arr;
}

void array_to_matrix(double *arr, Matrix *m) {
  for (uint32_t i = 0; i < m->rows; i++) {
    for (uint32_t j = 0; j < m->cols; j++) {
      m->data[i][j] = arr[i * m->cols + j];
    }
  }
}

Vector *cuda_vector_add_wrapper(Vector *a, Vector *b) {
  if (a->size != b->size) {
    fprintf(stderr, "Error: Vectors must be the same size for addition\n");
    return NULL;
  }

  Vector *result = vector_new(a->size);
  if (result == NULL) {
    return NULL;
  }

  double *d_a, *d_b, *d_result;
  cudaMalloc(&d_a, a->size * sizeof(double));
  cudaMalloc(&d_b, b->size * sizeof(double));
  cudaMalloc(&d_result, result->size * sizeof(double));

  cudaMemcpy(d_a, a->data, a->size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b->data, b->size * sizeof(double), cudaMemcpyHostToDevice);

  int block_size = 256;
  int num_blocks = (a->size + block_size - 1) / block_size;
  cuda_vector_add<<<num_blocks, block_size>>>(d_a, d_b, d_result, a->size);

  cudaMemcpy(result->data, d_result, result->size * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);

  return result;
}

Vector *cuda_vector_scale_wrapper(Vector *v, double scalar) {
  Vector *result = vector_new(v->size);
  if (result == NULL) {
    return NULL;
  }

  double *d_v, *d_result;
  cudaMalloc(&d_v, v->size * sizeof(double));
  cudaMalloc(&d_result, result->size * sizeof(double));

  cudaMemcpy(d_v, v->data, v->size * sizeof(double), cudaMemcpyHostToDevice);

  int block_size = 256;
  int num_blocks = (v->size + block_size - 1) / block_size;
  cuda_vector_scale<<<num_blocks, block_size>>>(d_v, scalar, d_result, v->size);

  cudaMemcpy(result->data, d_result, result->size * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_v);
  cudaFree(d_result);

  return result;
}

Matrix *cuda_matrix_add_wrapper(Matrix *a, Matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols) {
    fprintf(stderr, "Error: Matrices must be the same size for addition\n");
    return NULL;
  }

  Matrix *result = matrix_new(a->rows, a->cols);
  if (result == NULL) {
    return NULL;
  }

  double *h_a = matrix_to_array(a);
  double *h_b = matrix_to_array(b);
  double *h_result = (double *)malloc(a->rows * a->cols * sizeof(double));

  double *d_a, *d_b, *d_result;
  cudaMalloc(&d_a, a->rows * a->cols * sizeof(double));
  cudaMalloc(&d_b, b->rows * b->cols * sizeof(double));
  cudaMalloc(&d_result, result->rows * result->cols * sizeof(double));

  cudaMemcpy(d_a, h_a, a->rows * a->cols * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, b->rows * b->cols * sizeof(double),
             cudaMemcpyHostToDevice);

  dim3 block_size(16, 16);
  dim3 num_blocks((a->cols + block_size.x - 1) / block_size.x,
                  (a->rows + block_size.y - 1) / block_size.y);
  cuda_matrix_add<<<num_blocks, block_size>>>(d_a, d_b, d_result, a->rows,
                                              a->cols);

  cudaMemcpy(h_result, d_result, result->rows * result->cols * sizeof(double),
             cudaMemcpyDeviceToHost);

  array_to_matrix(h_result, result);

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
  free(h_a);
  free(h_b);
  free(h_result);

  return result;
}
