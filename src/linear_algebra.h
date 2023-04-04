#ifndef LA_H
#define LA_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>
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
    size_t size;
    double *data;
} Vector;

// Matrix struct

typedef struct {
    size_t rows;
    size_t cols;
    Vector **data;
} Matrix;

// Tensor struct

typedef struct {
    size_t rank;
    size_t rows;
    size_t cols;
    Matrix **data;
} Tensor;

// Vector functions
Vector *vector_new(size_t n);
void vector_free(Vector *v);
Vector *vector_copy(const Vector *v);
Vector *vector_add(const Vector *v1, const Vector *v2);
Vector *vector_sub(const Vector *v1, const Vector *v2);
Vector *vector_scale(const Vector *v, double s);
Vector *vector_multiply(const Vector *v1, const Vector *v2);
double vector_dot(const Vector *v1, const Vector *v2);
double vector_norm(const Vector *v);
Vector *vector_normalize(const Vector *v);
Vector *vector_cross(const Vector *v1, const Vector *v2);
Vector *vector_from_array(size_t n, const double *data);
double *vector_to_array(const Vector *v);

// Matrix functions
Matrix *matrix_new(size_t m, size_t n);
void matrix_free(Matrix *m);
Matrix *matrix_copy(const Matrix *m);
Matrix *matrix_add(const Matrix *m1, const Matrix *m2);
Matrix *matrix_sub(const Matrix *m1, const Matrix *m2);
Matrix *matrix_scale(const Matrix *m, double s);
Matrix *matrix_multiply(const Matrix *m1, const Matrix *m2);
Matrix *matrix_multiply_vector(const Matrix *m, const Vector *v);
Matrix *matrix_transpose(const Matrix *m);
void matrix_fill(Matrix *m, double s);
void matrix_set(Matrix *m, const Vector *v, size_t row);
void matrix_print(const Matrix *m);
Matrix *matrix_identity(size_t n);

// Tensor functions
Tensor *tensor_new(size_t num_matrices, size_t rows, size_t cols);
void tensor_free(Tensor *t);
Tensor *tensor_copy(const Tensor *t);
void tensor_insert(Tensor *t, const Matrix *m, size_t index);
Tensor *tensor_add(const Tensor *t1, const Tensor *t2);
Tensor *tensor_sub(const Tensor *t1, const Tensor *t2);
Tensor *tensor_scale(const Tensor *t, double s);
Tensor *tensor_dot(const Tensor *t1, const Tensor *t2);
Tensor *tensor_multiply(const Tensor *t1, const Tensor *t2);
Tensor *tensor_transpose(const Tensor *t);
void tensor_print(const Tensor *t);

#endif
