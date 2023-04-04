#include "linear_algebra.h"
#include <stdio.h>

// Vector functions
// -----------------------------------------------------------------------------
Vector *vector_new(int n) {
  Vector *v = malloc(sizeof(Vector));
  v->size = n;
  v->data = malloc(n * sizeof(double));
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

  for (int i = 0; i < v->size; i++) {
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

  for (int i = 0; i < a->size; i++) {
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

  for (int i = 0; i < a->size; i++) {
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

  for (int i = 0; i < v->size; i++) {
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

  for (int i = 0; i < a->size; i++) {
    result += a->data[i] * b->data[i];
  }

  return result;
}

double vector_norm(Vector *v) {
  double result = 0;

  for (int i = 0; i < v->size; i++) {
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

  for (int i = 0; i < v->size; i++) {
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

  for (int i = 0; i < a->size; i++) {
    result->data[i] = a->data[(i + 1) % 3] * b->data[(i + 2) % 3] -
                      a->data[(i + 2) % 3] * b->data[(i + 1) % 3];
  }

  return result;
}

Vector *vector_from_array(int size, double array[]) {
  Vector *result = vector_new(size);

  if (result == NULL) {
    fprintf(stderr, "Error: vector_from_array() failed to allocate memory");
    return NULL;
  }

  for (int i = 0; i < size; i++) {
    result->data[i] = array[i];
  }

  return result;
}

double *vector_to_array(Vector *v) {
  double *result = malloc(sizeof(double) * v->size);

  for (int i = 0; i < v->size; i++) {
    result[i] = v->data[i];
  }

  return result;
}

// Matrix functions
// -----------------------------------------------------------------------------
Matrix *matrix_new(int rows, int cols) {
  Matrix *result = (Matrix *)malloc(sizeof(Matrix));

  if (result == NULL) {
    fprintf(stderr, "Error: matrix_new() failed to allocate memory");
    return NULL;
  }

  result->rows = rows;
  result->cols = cols;

  result->data = (double **)malloc(rows * sizeof(double *));
  if (result->data == NULL) {
    fprintf(stderr, "Error: matrix_new() failed to allocate memory");
    return NULL;
  }

  for (int i = 0; i < rows; i++) {
    result->data[i] = (double *)malloc(cols * sizeof(double));
    if (result->data[i] == NULL) {
      fprintf(stderr,
              "Error: matrix_new() failed to allocate memory for row %d\n", i);
      return NULL;
    }
  }

  return result;
}

void matrix_free(Matrix *m) {
  if (m == NULL) {
    return;
  }

  for (int i = 0; i < m->rows; i++) {
    free(m->data[i]);
  }

  free(m->data);
  free(m);
}

void matrix_free_tensor(Matrix *m) {
  if (m == NULL) {
    return;
  }

  for (int i = 0; i < m->rows; i++) {
    free(m->data[i]);
  }
}

Matrix *matrix_copy(Matrix *m) {
  Matrix *result = matrix_new(m->rows, m->cols);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      result->data[i][j] = m->data[i][j];
    }
  }

  return result;
}

Matrix *matrix_add(Matrix *a, Matrix *b) {
  if (a->rows != b->rows && a->cols != b->cols) {
    fprintf(stderr,
            "Error: matrix_add() cannot add matrices of different sizes");
    return NULL;
  }

  Matrix *result = matrix_new(a->rows, a->cols);

  if (result == NULL) {
    fprintf(stderr,
            "Error: matrix_add() failed to allocate memory for result matrix");
    return NULL;
  }

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      result->data[i][j] = a->data[i][j] + b->data[i][j];
    }
  }

  return result;
}

Matrix *matrix_sub(Matrix *a, Matrix *b) {
  if (a->rows != b->rows && a->cols != b->cols) {
    fprintf(stderr,
            "Error: matrix_sub() cannot subtract matrices of different sizes");
    return NULL;
  }

  Matrix *result = matrix_new(a->rows, a->cols);

  if (result == NULL) {
    fprintf(stderr,
            "Error: matrix_sub() failed to allocate memory for result matrix");
    return NULL;
  }

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < a->cols; j++) {
      result->data[i][j] = a->data[i][j] - b->data[i][j];
    }
  }

  return result;
}

Matrix *matrix_scale(Matrix *m, double s) {
  Matrix *result = matrix_new(m->rows, m->cols);

  if (result == NULL) {
    fprintf(
        stderr,
        "Error: matrix_scale() failed to allocate memory for result matrix");
    return NULL;
  }

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      result->data[i][j] = m->data[i][j] * s;
    }
  }

  return result;
}

Matrix *matrix_multiply(Matrix *a, Matrix *b) {
  if (a->cols != b->rows) {
    fprintf(stderr, "Error: matrix_multiply() cannot multiply matrices of "
                    "incompatible sizes");
    return NULL;
  }

  Matrix *result = matrix_new(a->rows, b->cols);

  if (result == NULL) {
    fprintf(
        stderr,
        "Error: matrix_multiply() failed to allocate memory for result matrix");
    return NULL;
  }

  for (int i = 0; i < a->rows; i++) {
    for (int j = 0; j < b->cols; j++) {
      result->data[i][j] = 0;
      for (int k = 0; k < a->cols; k++) {
        result->data[i][j] += round(a->data[i][k] * b->data[k][j]);
      }
    }
  }

  return result;
}

Matrix *matrix_multiply_vector(Matrix *m, Vector *v) {
  if (m->cols != v->size) {
    fprintf(stderr, "Error: matrix_muliply_vector() cannot multiply matrix and "
                    "vector of incompatible sizes");
    return NULL;
  }

  Matrix *result = matrix_new(m->rows, 1);

  if (result == NULL) {
    fprintf(stderr, "Error: matrix_muliply_vector() failed to allocate memory "
                    "for result matrix");
    return NULL;
  }

  for (int i = 0; i < m->rows; i++) {
    result->data[i][0] = 0;
    for (int j = 0; j < m->cols; j++) {
      result->data[i][0] += m->data[i][j] * v->data[j];
    }
  }

  return result;
}

Matrix *matrix_transpose(Matrix *m) {
  Matrix *result = matrix_new(m->cols, m->rows);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      result->data[j][i] = m->data[i][j];
    }
  }

  return result;
}

void matrix_fill(Matrix *m, double value) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->data[i][j] = value;
    }
  }
}

void matrix_set(Matrix *m, double data[], int size) {
  if (size != m->rows * m->cols) {
    fprintf(
        stderr,
        "Error: matrix_set() cannot set matrix of size %d to array of size %d",
        m->rows * m->cols, size);
    return;
  }

  // Check if we have allocated memory for the matrix
  if (m->data == NULL) {
    fprintf(stderr, "Error: matrix_set() cannot set matrix with no allocated "
                    "memory for data");
    return;
  }

  for (int i = 0; i < size; i++) {
    int row = i / m->cols; // calculate the row index
    int col = i % m->cols; // calculate the column index
    m->data[row][col] = data[i];
  }
}

void matrix_print(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%f ", m->data[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

Matrix *matrix_identity(int size) {
  Matrix *result = matrix_new(size, size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      if (i == j) {
        result->data[i][j] = 1.0;
      } else {
        result->data[i][j] = 0.0;
      }
    }
  }

  return result;
}

// Tensor functions
// -----------------------------------------------------------------------------
Tensor *tensor_new(int rows, int cols, int rank) {
  Tensor *t = (Tensor *)malloc(sizeof(Tensor));

  if (t == NULL) {
    fprintf(stderr, "tensor_new: failed to allocate memory");
    return NULL;
  }

  t->rows = rows;
  t->cols = cols;
  t->rank = rank;

  t->data = (double ***)malloc(rank * sizeof(double **));
  if (t->data == NULL) {
    fprintf(stderr, "tensor_new: failed to allocate memory");
    return NULL;
  }

  for (int i = 0; i < rank; i++) {
    t->data[i] = (double **)malloc(rows * sizeof(double *));
  }

  for (int i = 0; i < rank; i++) {
    for (int j = 0; j < rows; j++) {
      t->data[i][j] = (double *)malloc(cols * sizeof(double));
    }
  }

  return t;
}

void tensor_insert(Tensor *t, Matrix *m, int index) {
  if (index >= t->rank) {
    fprintf(stderr, "tensor_insert: index out of bounds");
    return;
  }

  if (m->rows != t->rows || m->cols != t->cols) {
    fprintf(stderr, "tensor_insert: matrix size does not match tensor size");
    return;
  }

  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      t->data[index][i][j] = m->data[i][j];
    }
  }
}

void tensor_free(Tensor *t) {
  for (int i = 0; i < t->rank; i++) {
    for (int j = 0; j < t->rows; j++) {
      free(t->data[i][j]);
    }
  }

  for (int i = 0; i < t->rank; i++) {
    free(t->data[i]);
  }

  free(t->data);
  free(t);
}

Tensor *tensor_copy(Tensor *src) {
  Tensor *dest = tensor_new(src->rows, src->cols, src->rank);

  for (int i = 0; i < src->rank; i++) {
    for (int j = 0; j < src->rows; j++) {
      for (int k = 0; k < src->cols; k++) {
        dest->data[i][j][k] = src->data[i][j][k];
      }
    }
  }

  return dest;
}

void tensor_print(Tensor *t) {
  for (int i = 0; i < t->rank; i++) {
    printf("Matrix %d\n", i);
    for (int j = 0; j < t->rows; j++) {
      for (int k = 0; k < t->cols; k++) {
        printf("%f ", t->data[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

Tensor *tensor_add(Tensor *t1, Tensor *t2) {
  if (t1->rank != t2->rank) {
    fprintf(stderr, "tensor_add: tensors must have the same rank");
    return NULL;
  }

  Tensor *result = tensor_new(t1->rank, t1->rows, t1->cols);
  if (result == NULL) {
    fprintf(stderr, "tensor_add: failed to allocate memory");
    return NULL;
  }

  for (int i = 0; i < t1->rank; i++) {
    for (int j = 0; j < t1->rows; j++) {
      for (int k = 0; k < t1->cols; k++) {
        result->data[i][j][k] = t1->data[i][j][k] + t2->data[i][j][k];
      }
    }
  }

  return result;
}

Tensor *tensor_sub(Tensor *t1, Tensor *t2) {
  if (t1->rank != t2->rank) {
    fprintf(stderr, "tensor_sub: tensors must have the same rank");
    return NULL;
  }

  Tensor *result = tensor_new(t1->rank, t1->rows, t1->cols);
  if (result == NULL) {
    fprintf(stderr, "tensor_sub: failed to allocate memory");
    return NULL;
  }

  for (int i = 0; i < t1->rank; i++) {
    for (int j = 0; j < t1->rows; j++) {
      for (int k = 0; k < t1->cols; k++) {
        result->data[i][j][k] = t1->data[i][j][k] - t2->data[i][j][k];
      }
    }
  }

  return result;
}

// Tensor *tensor_scale(Tensor *t, double scale) {
//   Tensor *result = tensor_new(t->rank, t->rows, t->cols);
//   if (result == NULL) {
//     fprintf(stderr, "tensor_scale: failed to allocate memory");
//     return NULL;
//   }
//
//   for (int i = 0; i < t->rank; i++) {
//     Matrix *temp;
//     temp = matrix_scale(&(t->data[i]), scale);
//
//     if (temp == NULL) {
//       printf("Error: unable to scale tensor\n");
//       return NULL;
//     }
//
//     tensor_insert(result, temp, i);
//   }
//
//   return result;
// }

// Tensor* tensor_dot(Tensor *t1, Tensor *t2) {
//   if (t1->rank != t2->rank) {
//     // handle error
//     printf("Error: tensors must be the same size\n");
//     return NULL;
//   }
//
//   Tensor *result = tensor_new(t1->rank, t1->rows, t1->cols);
//   if (result == NULL) {
//     // handle error
//     printf("Error: unable to allocate memory for tensor\n");
//     return NULL;
//   }
//
//   for (int i = 0; i < t1->rank; i++) {
//     Matrix *temp;
//     temp = matrix_dot(&(t1->data[i]), &(t2->data[i]));
//     if (temp == NULL) {
//       // handle error
//       printf("Error: unable to add tensors\n");
//       return NULL;
//     }
//     tensor_insert(result, temp, i);
//   }
//
//   return result;
// }

// Tensor *tensor_multiply(Tensor *t1, Tensor *t2) {
//   if (t1->rank != t2->rank) {
//     fprintf(stderr, "tensor_multiply: tensors must be the same size");
//     return NULL;
//   }
//
//   Tensor *result = tensor_new(t1->rank, t1->rows, t1->cols);
//   if (result == NULL) {
//     fprintf(stderr, "tensor_multiply: failed to allocate memory");
//     return NULL;
//   }
//
//   for (int i = 0; i < t1->rank; i++) {
//     Matrix *temp;
//     temp = matrix_multiply(&(t1->data[i]), &(t2->data[i]));
//     if (temp == NULL) {
//       fprintf(stderr, "tensor_multiply: unable to multiply tensors");
//       return NULL;
//     }
//     tensor_insert(result, temp, i);
//   }
//
//   return result;
// }
