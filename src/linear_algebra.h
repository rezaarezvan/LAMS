#ifndef LA_H
#define LA_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * A Linear Algebra library for C
 *
 * Using general structs to represent:
 * vectors, matrices and tensors of any dimension
 *
 */

// Vector struct

typedef struct {
  int size;
  double *data;
} Vector;

// Matrix struct

typedef struct {
  int rows, cols;
  double **data;
} Matrix;

// Tensor struct

typedef struct {
  int rank, rows, cols;
  double ***data;
} Tensor;

// Vector functions
Vector *vector_new(int n);
void vector_free(Vector *v);
Vector *vector_copy(Vector *v);
Vector *vector_add(Vector *v1, Vector *v2);
Vector *vector_sub(Vector *v1, Vector *v2);
Vector *vector_scale(Vector *v, double s);
double vector_dot(Vector *v1, Vector *v2);
double vector_norm(Vector *v);
Vector *vector_normalize(Vector *v);
Vector *vector_cross(Vector *v1, Vector *v2);
Vector *vector_from_array(int n, double *data);
double *vector_to_array(Vector *v);

// Matrix functions
Matrix *matrix_new(int m, int n);
void matrix_free(Matrix *m);
Matrix *matrix_copy(Matrix *m);
Matrix *matrix_add(Matrix *m1, Matrix *m2);
Matrix *matrix_sub(Matrix *m1, Matrix *m2);
Matrix *matrix_scale(Matrix *m, double s);
Matrix *matrix_multiply(Matrix *m1, Matrix *m2);
Matrix *matrix_multiply_vector(Matrix *m, Vector *v);
Matrix *matrix_transpose(Matrix *m);
void matrix_fill(Matrix *m, double s);
void matrix_set(Matrix *m, double data[], int size);
void matrix_print(Matrix *m);
Matrix *matrix_identity(int n);
Matrix *matrix_solve(Matrix *A, Matrix *b);
Matrix *matrix_solve_lu(Matrix *A, Matrix *b);

// Tensor functions
Tensor *tensor_new(int num_matrices, int rows, int cols);
void tensor_insert(Tensor *t, Matrix *m, int index);
void tensor_free(Tensor *t);
Tensor *tensor_copy(Tensor *t);
void tensor_print(Tensor *t);
Tensor *tensor_add(Tensor *t1, Tensor *t2);
Tensor *tensor_sub(Tensor *t1, Tensor *t2);
Tensor *tensor_scale(Tensor *t, double s);
Tensor *tensor_dot(Tensor *t1, Tensor *t2);
Tensor *tensor_multiply(Tensor *t1, Tensor *t2);
Tensor *tensor_transpose(Tensor *t);

#endif
