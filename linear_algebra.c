#include "linear_algebra.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Vector functions

Vector* vector_new(int n) {
    Vector* v = malloc(sizeof(Vector));
    v->size = n;
    v->data = malloc(n * sizeof(double));
    return v;
}

void vector_free(Vector* v) {
    free(v->data);
    free(v);
}

Vector* vector_copy(Vector* v) {
    Vector* v_copy = vector_new(v->size);
    for (int i = 0; i < v->size; i++) {
        v_copy->data[i] = v->data[i];
    }
    return v_copy;
}

Vector* vector_add(Vector *a, Vector *b) {
    Vector *result = malloc(sizeof(Vector));

    if(a->size != b->size) {
        printf("Error: vector sizes do not match");
        return NULL;
    }

    for(int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}

Vector* vector_sub(Vector *a, Vector *b) {
    Vector *result = malloc(sizeof(Vector));

    if(a->size != b->size) {
        printf("Error: vector sizes do not match");
        return NULL;
    }

    for(int i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}

Vector* vector_scale(Vector *v, double c) {
    Vector *result = malloc(sizeof(Vector));

    for(int i = 0; i < v->size; i++) {
        result->data[i] = v->data[i] * c;
    }

    return result;
}

double vector_dot(Vector *a, Vector *b) {
    double result = 0;

    if(a->size != b->size) {
        printf("Error: vector sizes do not match");
        return 0;
    }

    for(int i = 0; i < a->size; i++) {
        result += a->data[i] * b->data[i];
    }

    return result;
}

double vector_norm(Vector *v) {
    double result = 0;

    for(int i = 0; i < v->size; i++) {
        result += v->data[i] * v->data[i];
    }

    return sqrt(result);
}

Vector* vector_normalize(Vector *v) {
    Vector *result = malloc(sizeof(Vector));

    double norm = vector_norm(v);

    for(int i = 0; i < v->size; i++) {
        result->data[i] = v->data[i] / norm;
    }

    return result;
}

Vector* vector_cross(Vector *a, Vector *b) {
    Vector *result = malloc(sizeof(Vector));

    if(a->size != b->size) {
        printf("Error: vector sizes do not match");
        return NULL;
    }
    
    for(int i = 0; i < a->size; i++) {
      result->data[i] = a->data[(i+1)%3] * b->data[(i+2)%3] - a->data[(i+2)%3] * b->data[(i+1)%3];
    }

    return result;
}

Vector* vector_from_array(int size, double *array) {
    Vector *result = malloc(sizeof(Vector));

    for(int i = 0; i < size; i++) {
        result->data[i] = array[i];
    }

    return result;
}

double* vector_to_array(Vector *v) {
    double result[v->size];

    for(int i = 0; i < v->size; i++) {
        result[i] = v->data[i];
    }

    return result;
}

// Matrix functions
// ----------------

Matrix* matrix_new(int rows, int cols) {
    Matrix *result = malloc(sizeof(Matrix));

    result->rows = rows;
    result->cols = cols;

    return result;
}

void matrix_free(Matrix *m) {
    free(m);
}

Matrix* matrix_copy(Matrix *m) {
    Matrix *result = matrix_new(m->rows, m->cols);

    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            result->data[i][j] = m->data[i][j];
        }
    }

    return result;
}

Matrix* matrix_add(Matrix *a, Matrix *b) {
    Matrix *result = matrix_new(a->rows, a->cols);

    if(a->rows != b->rows && a->cols != b->cols) {
        printf("Error: matrix sizes do not match");
        return NULL;
    }

    for(int i = 0; i < a->rows; i++) {
        for(int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] + b->data[i][j];
        }
    }

    return result;
}

Matrix* matrix_sub(Matrix *a, Matrix *b) {
    Matrix *result = matrix_new(a->rows, a->cols);

    if(a->rows != b->rows && a->cols != b->cols) {
        printf("Error: matrix sizes do not match");
        return NULL;
    }

    for(int i = 0; i < a->rows; i++) {
        for(int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }

    return result;
}

Matrix* matrix_scale(Matrix *m, double s) {
    Matrix *result = matrix_new(m->rows, m->cols);

    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            result->data[i][j] = m->data[i][j] * s;
        }
    }

    return result;
}

Matrix* matrix_multiply(Matrix *a, Matrix *b) {
    Matrix *result = matrix_new(a->rows, b->cols);

    if(a->cols != b->rows) {
        printf("Error: matrix sizes do not match");
        return NULL;
    }

    for(int i = 0; i < a->rows; i++) {
        for(int j = 0; j < b->cols; j++) {
            for(int k = 0; k < a->cols; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }

    return result;
}

Matrix* matrix_muliply_vector(Matrix *m, Vector *v) {
    Matrix *result = matrix_new(m->rows, 1);

    if(m->cols != v->size) {
        printf("Error: matrix sizes do not match");
        return NULL;
    }

    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            result->data[i][0] += m->data[i][j] * v->data[j];
        }
    }

    return result;
}

Matrix* matrix_transpose(Matrix *m) {
    Matrix *result = matrix_new(m->cols, m->rows);

    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            result->data[j][i] = m->data[i][j];
        }
    }

    return result;
}

void fill_matrix(Matrix *m, double value) {
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            m->data[i][j] = value;
        }
    }
}

void set_matrix(Matrix *m, double data[], int size) {
    // check if the matrix sizes match
    if(size != m->rows * m->cols) {
        printf("Error: matrix sizes do not match\n");
        return;
    }

    // allocate memory for the matrix data
    m->data = malloc(m->rows * sizeof(double*));
    if (m->data == NULL) {
        // handle memory allocation error
        printf("Error: unable to allocate memory for matrix data\n");
        return;
    }

    for (int i = 0; i < m->rows; i++) {
        m->data[i] = malloc(m->cols * sizeof(double));
        if (m->data[i] == NULL) {
            // handle memory allocation error
            printf("Error: unable to allocate memory for matrix data\n");
            return;
        }
    }

    // copy the data from the input array to the matrix
    for (int i = 0; i < size; i++) {
        int row = i / m->cols;   // calculate the row index
        int col = i % m->cols;   // calculate the column index
        m->data[row][col] = data[i];
    }
}
void print_matrix(Matrix *m) {
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            printf("%f ", m->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix* matrix_from_array(int m, int n, double *data) {
    Matrix *result = matrix_new(m, n);

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            result->data[i][j] = data[i * n + j];
        }
    }

    return result;
}

Matrix* matrix_identity(int n) {
    Matrix *result = matrix_new(n, n);

    for(int i = 0; i < n; i++) {
        result->data[i][i] = 1;
    }

    return result;
}

Matrix* matrix_rotate_x(double angle) {
    Matrix *result = matrix_identity(4);

    result->data[1][1] = cos(angle);
    result->data[1][2] = -sin(angle);
    result->data[2][1] = sin(angle);
    result->data[2][2] = cos(angle);

    return result;
}

Matrix* matrix_rotate_y(double angle) {
    Matrix *result = matrix_identity(4);

    result->data[0][0] = cos(angle);
    result->data[0][2] = sin(angle);
    result->data[2][0] = -sin(angle);
    result->data[2][2] = cos(angle);

    return result;
}

Matrix* matrix_rotate_z(double angle) {
    Matrix *result = matrix_identity(4);

    result->data[0][0] = cos(angle);
    result->data[0][1] = -sin(angle);
    result->data[1][0] = sin(angle);
    result->data[1][1] = cos(angle);

    return result;
}

Matrix* matrix_translation(double x, double y, double z) {
    Matrix *result = matrix_identity(4);

    result->data[0][3] = x;
    result->data[1][3] = y;
    result->data[2][3] = z;

    return result;
}

// Solving linear system functions
// Solves Ax = b
// A is a square matrix
// b is a vector

Matrix* matrix_solve(Matrix *A, Matrix *b) {
    int n = A->cols;
    Matrix *x = matrix_new(n, 1);

    // Copy A and b to temporary matrices
    Matrix *A_temp = matrix_copy(A);
    Matrix *b_temp = matrix_copy(b);

    // Forward elimination
    for(int i = 0; i < n - 1; i++) {
        // Find the row with the largest pivot
        int max_row = i;
        for(int j = i + 1; j < n; j++) {
            if(fabs(A_temp->data[j][i]) > fabs(A_temp->data[max_row][i])) {
                max_row = j;
            }
        }

        // Swap the rows
        for(int j = i; j < n; j++) {
            double temp = A_temp->data[i][j];
            A_temp->data[i][j] = A_temp->data[max_row][j];
            A_temp->data[max_row][j] = temp;
        }

        double temp = b_temp->data[i][0];
        b_temp->data[i][0] = b_temp->data[max_row][0];
        b_temp->data[max_row][0] = temp;

        // Eliminate the lower rows
        for(int j = i + 1; j < n; j++) {
            double c = A_temp->data[j][i] / A_temp->data[i][i];
            for(int k = i; k < n; k++) {
                A_temp->data[j][k] -= A_temp->data[i][k] * c;
            }
            b_temp->data[j][0] -= b_temp->data[i][0] * c;
        }
    }

    // Back substitution
    for(int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for(int j = i + 1; j < n; j++) {
            sum += A_temp->data[i][j] * x->data[j][0];
        }
        x->data[i][0] = (b_temp->data[i][0] - sum) / A_temp->data[i][i];
    }

    matrix_free(A_temp);
    matrix_free(b_temp);

    return x;
}

Matrix* matrix_solve_lu(Matrix *A, Matrix *b) {
    int n = A->cols;

    // check if the dimensions of the input matrices are compatible
    if (A->cols != b->rows) {
        printf("Error: matrix dimensions are not compatible\n");
        return NULL;
    }

    Matrix *x = matrix_new(n, 1);

    // Copy A and b to temporary matrices
    Matrix *A_temp = matrix_copy(A);
    if (A_temp == NULL) {
        // handle error
        printf("Error: unable to copy matrix A\n");
        return NULL;
    }
    Matrix *b_temp = matrix_copy(b);
    if (b_temp == NULL) {
        // handle error
        printf("Error: unable to copy matrix b\n");
        return NULL;
    }

    // LU decomposition
    for(int i = 0; i < n - 1; i++) {
        // Eliminate the lower rows
        for(int j = i + 1; j < n; j++) {
            double c = A_temp->data[j][i] / A_temp->data[i][i];
            for(int k = i; k < n; k++) {
                A_temp->data[j][k] -= A_temp->data[i][k] * c;
            }
            b_temp->data[j][0] -= b_temp->data[i][0] * c;
        }
    }

    // Back substitution
    for(int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for(int j = i + 1; j < n; j++) {
            sum += A_temp->data[i][j] * x->data[j][0];
        }
        x->data[i][0] = (b_temp->data[i][0] - sum) / A_temp->data[i][i];
    }

    matrix_free(A_temp);
    matrix_free(b_temp);

    return x;
}

int main() {
    // Example 1
    // Matrix A = [1 2 3; 4 5 6; 7 8 9]
    // Matrix B = [3; 3; 4]
    double data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int size = sizeof(data1) / sizeof(data1[0]);
    printf("size = %d\n", size);

    Matrix* A = matrix_new(3, 3);
    set_matrix(A, data1, size);
    print_matrix(A);
    
    // A random 3x3 matrix
    double data2[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    size = sizeof(data2) / sizeof(data2[0]);
    printf("size = %d\n", size);

    Matrix* B = matrix_new(3, 3);
    set_matrix(B, data2, size);
    print_matrix(B);

    Matrix* res = matrix_multiply(A, B);
    print_matrix(res);
    
    return 1;
}
