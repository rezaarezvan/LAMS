#include "linear_algebra.h"
#include <stdio.h>

// Vector functions
// -----------------------------------------------------------------------------
Vector *vector_new(uint32_t n) {
  Vector *v = malloc(sizeof(Vector));
  v->size = n;
  v->data = malloc(sizeof *v->data * n);
  return v;
}

void vector_free(Vector *v) {
  free(v->data);
  free(v);
  v = NULL;
}

Vector *vector_copy(Vector *v) {
  Vector *v_copy = vector_new(v->size);

  if (v_copy == NULL) {
    fprintf(stderr, "Error: vector_copy() failed to allocate memory");
    return NULL;
  }

  for (uint32_t i = 0; i < v->size; i++) {
    v_copy->data[i] = v->data[i];
  }

  return v_copy;
}

Vector *vector_add(Vector *a, Vector *b) {
  if (a->size != b->size) {
    fprintf(stderr, "Error: vector_add() vectors must be the same size");
    return NULL;
  }

  Vector *result = vector_new(a->size);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_add() failed to allocate memory");
    return NULL;
  }

  for (uint32_t i = 0; i < a->size; i++) {
    result->data[i] = a->data[i] + b->data[i];
  }

  return result;
}

Vector *vector_sub(Vector *a, Vector *b) {
  if (a->size != b->size) {
    fprintf(stderr, "Error: vector_sub() vectors must be the same size");
    return NULL;
  }

  Vector *result = vector_new(a->size);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_sub() failed to allocate memory");
    return NULL;
  }

  for (uint32_t i = 0; i < a->size; i++) {
    result->data[i] = a->data[i] - b->data[i];
  }

  return result;
}

Vector *vector_scale(Vector *v, double c) {
  Vector *result = vector_new(v->size);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_scale() failed to allocate memory");
    return NULL;
  }

  for (uint32_t i = 0; i < v->size; i++) {
    result->data[i] = v->data[i] * c;
  }

  return result;
}

double vector_dot(Vector *a, Vector *b) {
  double result = 0;

  if (a->size != b->size) {
    fprintf(stderr, "Error: vector_dot() vectors must be the same size");
    return 0;
  }

  for (uint32_t i = 0; i < a->size; i++) {
    result += a->data[i] * b->data[i];
  }

  return result;
}

double vector_norm(Vector *v) {
  double result = 0;

  for (uint32_t i = 0; i < v->size; i++) {
    result += v->data[i] * v->data[i];
  }

  return sqrt(result);
}

Vector *vector_normalize(Vector *v) {
  Vector *result = vector_new(v->size);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_normalize() failed to allocate memory");
    return NULL;
  }

  double norm = vector_norm(v);

  for (uint32_t i = 0; i < v->size; i++) {
    result->data[i] = v->data[i] / norm;
  }

  return result;
}

Vector *vector_cross(Vector *a, Vector *b) {
  if (a->size != b->size) {
    fprintf(stderr, "Error: vector_cross() vectors must be the same size");
    return NULL;
  }

  Vector *result = vector_new(3);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_cross() failed to allocate memory");
    return NULL;
  }

  for (uint32_t i = 0; i < a->size; i++) {
    result->data[i] = a->data[(i + 1) % 3] * b->data[(i + 2) % 3] -
                      a->data[(i + 2) % 3] * b->data[(i + 1) % 3];
  }

  return result;
}

Vector *vector_from_array(uint32_t size, double array[]) {
  Vector *result = vector_new(size);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_from_array() failed to allocate memory");
    return NULL;
  }

  for (uint32_t i = 0; i < size; i++) {
    result->data[i] = array[i];
  }

  return result;
}

double *vector_to_array(Vector *v) {
  double *result = malloc(v->size * sizeof *result);

  for (uint32_t i = 0; i < v->size; i++) {
    result[i] = v->data[i];
  }

  return result;
}

// Matrix functions
// -----------------------------------------------------------------------------
Matrix *matrix_new(uint32_t m, uint32_t n) {
  Matrix *matrix = malloc(sizeof(Matrix));
  matrix->rows = m;
  matrix->cols = n;
  matrix->data = malloc(m * sizeof(Vector *));
  for (uint32_t i = 0; i < m; i++) {
    matrix->data[i] = vector_new(n);
  }
  return matrix;
}

void matrix_free(Matrix *m) {
  for (uint32_t i = 0; i < m->rows; i++) {
    vector_free(m->data[i]);
  }
  free(m->data);
  free(m);
}

// Tensor functions
// -----------------------------------------------------------------------------
Tensor *tensor_new(uint32_t rank, uint32_t num_matrices, uint32_t rows,
                   uint32_t cols) {
  Tensor *tensor = malloc(sizeof(Tensor));
  tensor->rank = rank;
  tensor->num_matrices = num_matrices;
  tensor->data = malloc(num_matrices * sizeof(Matrix *));
  for (uint32_t i = 0; i < num_matrices; i++) {
    tensor->data[i] = matrix_new(rows, cols);
  }
  return tensor;
}

void tensor_free(Tensor *t) {
  for (uint32_t i = 0; i < t->num_matrices; i++) {
    matrix_free(t->data[i]);
  }
  free(t->data);
  free(t);
}
