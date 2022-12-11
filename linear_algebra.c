#include "linear_algebra.h"

// Vector functions
// -----------------------------------------------------------------------------

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
// -----------------------------------------------------------------------------

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

// Transfers data from matrix A to matrix B
// Needs to check if A is NULL/empty
// If A is empty then return ERROR, cannot copy empty matrix
// If B is empty then create a new matrix with the same dimensions as A
// If B is not empty then check if the dimensions of A and B are the same
// If the dimensions are the same then copy the data from A to B
// This still segfaults, why?
// As soon as I just acces b->data[0][0] it segfaults
// I think it has something to do with the fact that I'm not allocating memory for b->data
// But I don't know how to do that

Matrix* matrix_transfer(Matrix *src, Matrix *dest) {
    if (src->rows == 0 || src->cols == 0) {
        // handle error
        printf("Error: matrix A is empty\n");
        return;
    }

    if (dest->rows == 0 || dest->cols == 0) {
        dest = matrix_new(src->rows, src->cols);
        // allocate memory for the matrix data
        dest->data = malloc(dest->rows * sizeof(double*));
        if (dest->data == NULL) {
            // handle memory allocation error
            printf("Error: unable to allocate memory for matrix data\n");
            return;
        }

        for (int i = 0; i < dest->rows; i++) {
            dest->data[i] = malloc(dest->cols * sizeof(double));
            if (dest->data[i] == NULL) {
                // handle memory allocation error
                printf("Error: unable to allocate memory for matrix data\n");
                return;
            }
        }
    }

    if (src->rows != dest->rows || src->cols != dest->cols) {
        // handle error
        printf("Error: matrix A and B have different dimensions\n");
        return;
    }

    for (int i = 0; i < src->rows; i++) {
        for (int j = 0; j < src->cols; j++) {
            dest->data[i][j] = src->data[i][j];
        }
    }

    return dest;
}

// Tensor functions
// -----------------------------------------------------------------------------

Tensor* tensor_new(int rows, int cols, int rank) {
    Tensor *t = malloc(sizeof(Tensor));
    if (t == NULL) {
        // handle error
        printf("Error: unable to allocate memory for tensor\n");
        return NULL;
    }

    t->rows = rows;
    t->cols = cols;
    t->rank = rank;

    t->data = malloc(rank * sizeof(Matrix*));
    if (t->data == NULL) {
        // handle error
        printf("Error: unable to allocate memory for tensor data\n");
        return NULL;
    }

    for(int i = 0; i < rank; i++) {
        t->data[i] = *matrix_new(0, 0);
        if (&(t->data[i]) == NULL) {
            // handle error
            printf("Error: unable to allocate memory for tensor data\n");
            return NULL;
        }
    }

    return t;
}

void tensor_insert(Tensor *t, Matrix *m, int index) {
    if (index >= t->rank) {
        // handle error
        printf("Error: index out of bounds\n");
        return;
    }

    t->data[index].rows = m->rows;
    t->data[index].cols = m->cols;
    t->data[index] = *m;
}

void tensor_free(Tensor *t) {
    for(int i = 0; i < t->rank; i++) {
        matrix_free(&(t->data[i]));
    }
    free(t->data);
    free(t);
}

// Copies tensor, src, to tensor, dest.
// If dest is NULL, a new tensor is created.
// If dest is not NULL, it must be the same size as src.
// If an error occurs, returns NULL.
// -----------------------------------------------------------------------------

// This program segfaults if the tensor is not initialized to 0. Why?
// -----------------------------------------------------------------------------
void tensor_copy(Tensor *src, Tensor *dest) {
    if (dest == NULL) {
        dest = tensor_new(src->rank, src->rows, src->cols);
    }

    if (dest->rank != src->rank) {
        // handle error
        printf("Error: tensors must be the same size\n");
        return;
    }

    for(int i = 0; i < src->rank; i++) {
        Matrix* temp;
        temp = matrix_transfer(&(src->data[i]), &(dest->data[i]));
        if (temp == NULL) {
            // handle error
            printf("Error: unable to copy tensor\n");
            return;
        }
        dest->data[i] = *temp;
    }
}


// Print function for a tensor
// -----------------------------------------------------------------------------

void tensor_print(Tensor *t) {
    for(int i = 0; i < t->rank; i++) {
        printf("Tensor rank %d:\n", i);
        if (t->data[i].rows == 0 || t->data[i].cols == 0) {
            printf("Empty matrix\n");
        } else {
            print_matrix(&(t->data[i]));
        }
    }
}

int main() {
    double data1[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int size = sizeof(data1) / sizeof(data1[0]);

    Matrix* A = matrix_new(3, 3);
    set_matrix(A, data1, size);
    
    // A random 3x3 matrix
    double data2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18};
    size = sizeof(data2) / sizeof(data2[0]);

    Matrix* B = matrix_new(3, 3);
    set_matrix(B, data2, size);
    
    Tensor* C = tensor_new(2, 2, 2);
    tensor_insert(C, A, 0);
    tensor_insert(C, B, 1);
    printf("Tensor C:\n");
    tensor_print(C);
        
    Tensor* D = tensor_new(2, 2, 2);
    tensor_copy(C, D);
    printf("Tensor D:\n");
    tensor_print(D);
    C->data[0].data[0][0] = 100;
    printf("Mutated Tensor C:\n");
    tensor_print(C);
    printf("Tensor D:\n");
    tensor_print(D);
    
    return 1;
}
