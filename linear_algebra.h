#ifndef LA_H
#define LA_H

// A linear algebra library for C
// Using general structs to represent vectors and matrices of any dimension
// and type

typedef struct {
    int size;
    double *data;
} Vector;

typedef struct {
    int rows, cols;
    double **data;
} Matrix;

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
Vector *matrix_multiply_vector(Matrix *m, Vector *v);
Matrix *matrix_transpose(Matrix *m);
Matrix *matrix_from_array(int m, int n, double *data);
Matrix *matrix_identity(int n);
Matrix *matrix_rotation_x(double theta);
Matrix *matrix_rotation_y(double theta);
Matrix *matrix_rotation_z(double theta);
Matrix *matrix_rotation(Vector *axis, double theta);
Matrix *matrix_translation(double x, double y, double z);

// Functions to solve linear systems of form Ax = b
// using Gaussian elimination with partial pivoting
// and back substitution
Matrix *matrix_solve(Matrix *A, Matrix *b);
Matrix *matrix_solve_lu(Matrix *A, Matrix *b);
Matrix *matrix_lu(Matrix *A);

#endif
