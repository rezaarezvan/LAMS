#include "../src/cuda_linear_algebra.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

#define EPSILON 1e-6

int double_equals(double a, double b) { return fabs(a - b) < EPSILON; }

void test_cuda_vector_add() {
  const uint32_t size = 1000000;
  Vector *a = vector_new(size);
  Vector *b = vector_new(size);

  for (uint32_t i = 0; i < size; i++) {
    a->data[i] = i;
    b->data[i] = 2 * i;
  }

  Vector *result = cuda_vector_add_wrapper(a, b);

  for (uint32_t i = 0; i < size; i++) {
    assert(double_equals(result->data[i], 3 * i));
  }

  vector_free(a);
  vector_free(b);
  vector_free(result);
  printf("CUDA vector addition test passed\n");
}

void test_cuda_vector_scale() {
  const uint32_t size = 1000000;
  Vector *v = vector_new(size);
  double scalar = 2.5;

  for (uint32_t i = 0; i < size; i++) {
    v->data[i] = i;
  }

  Vector *result = cuda_vector_scale_wrapper(v, scalar);

  for (uint32_t i = 0; i < size; i++) {
    assert(double_equals(result->data[i], i * scalar));
  }

  vector_free(v);
  vector_free(result);
  printf("CUDA vector scaling test passed\n");
}

void test_cuda_matrix_add() {
  const uint32_t rows = 1000, cols = 1000;
  Matrix *a = matrix_new(rows, cols);
  Matrix *b = matrix_new(rows, cols);

  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      a->data[i][j] = i + j;
      b->data[i][j] = 2 * (i + j);
    }
  }

  Matrix *result = cuda_matrix_add_wrapper(a, b);

  for (uint32_t i = 0; i < rows; i++) {
    for (uint32_t j = 0; j < cols; j++) {
      assert(double_equals(result->data[i][j], 3 * (i + j)));
    }
  }

  matrix_free(a);
  matrix_free(b);
  matrix_free(result);
  printf("CUDA matrix addition test passed\n");
}

int main() {
  test_cuda_vector_add();
  test_cuda_vector_scale();
  test_cuda_matrix_add();

  printf("All CUDA tests passed!\n");
  return 0;
}
