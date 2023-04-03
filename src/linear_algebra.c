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
  if (v == NULL) {
    return;
  }

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

void vector_print(Vector *v) {
  printf("[");
  for (int i = 0; i < v->size; i++) {
    printf("%f", v->data[i]);
    if (i < v->size - 1) {
      printf(", ");
    }
  }
  printf("]");
}

// Matrix functions
// -----------------------------------------------------------------------------
Matrix *matrix_new(int rows, int cols) {
  Matrix *m = malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->data = malloc(rows * sizeof(Vector));

  for (int i = 0; i < rows; i++) {
    m->data[i] = vector_new(cols);
  }

  return m;
}

void matrix_free(Matrix *m) {
  if (m == NULL) {
    return;
  }

  for (int i = 0; i < m->rows; i++) {
    if (&(m->data[i]) != NULL) {
      vector_free(m->data[i]);
    }
  }

  free(m->data);
  free(m);
  m = NULL;
}

// Copies an existing matrix to a new matrix
Matrix *matrix_copy(Matrix *m) {
  Matrix *result = matrix_new(m->rows, m->cols);

  if (result == NULL) {
    fprintf(stderr, "Error: matrix_copy() failed to allocate memory");
    return NULL;
  }

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      result->data[i]->data[j] = m->data[i]->data[j];
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
      result->data[i]->data[j] = a->data[i]->data[j] + b->data[i]->data[j];
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
      result->data[i]->data[j] = a->data[i]->data[j] - b->data[i]->data[j];
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
      result->data[i]->data[j] = m->data[i]->data[j] * s;
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
      result->data[i]->data[j] = 0;
      for (int k = 0; k < a->cols; k++) {
        result->data[i]->data[j] += a->data[i]->data[k] * b->data[k]->data[j];
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
    result->data[i]->data[0] = 0;
    for (int j = 0; j < m->cols; j++) {
      result->data[i]->data[0] += roundf(m->data[i]->data[j] * v->data[j]);
    }
  }

  return result;
}

Matrix *matrix_transpose(Matrix *m) {
  Matrix *result = matrix_new(m->cols, m->rows);

  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      result->data[j]->data[i] = m->data[i]->data[j];
    }
  }

  return result;
}

void matrix_fill(Matrix *m, double value) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      m->data[i]->data[j] = value;
    }
  }
}

void matrix_set(Matrix *m, Vector *v, int row) {
  for (int i = 0; i < m->cols; i++) {
    m->data[row]->data[i] = v->data[i];
  }
}

void matrix_print(Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%f ", m->data[i]->data[j]);
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
        result->data[i]->data[j] = 1.0;
      } else {
        result->data[i]->data[j] = 0.0;
      }
    }
  }

  return result;
}

// TODO: Fix
// Matrix *matrix_solve(Matrix *A, Matrix *b) {
//   int n = A->cols;
//   Matrix *x = matrix_new(n, 1);
//
//   // Copy A and b to temporary matrices
//   Matrix *A_temp = matrix_copy(A);
//   Matrix *b_temp = matrix_copy(b);
//
//   // Forward elimination
//   for (int i = 0; i < n - 1; i++) {
//     // Find the row with the largest pivot
//     int max_row = i;
//     for (int j = i + 1; j < n; j++) {
//       if (fabs(A_temp->data[j][i]) > fabs(A_temp->data[max_row][i])) {
//         max_row = j;
//       }
//     }
//
//     // Swap the rows
//     for (int j = i; j < n; j++) {
//       double temp = A_temp->data[i][j];
//       A_temp->data[i][j] = A_temp->data[max_row][j];
//       A_temp->data[max_row][j] = temp;
//     }
//
//     double temp = b_temp->data[i][0];
//     b_temp->data[i][0] = b_temp->data[max_row][0];
//     b_temp->data[max_row][0] = temp;
//
//     // Eliminate the lower rows
//     for (int j = i + 1; j < n; j++) {
//       double c = A_temp->data[j][i] / A_temp->data[i][i];
//       for (int k = i; k < n; k++) {
//         A_temp->data[j][k] -= A_temp->data[i][k] * c;
//       }
//       b_temp->data[j][0] -= b_temp->data[i][0] * c;
//     }
//   }
//
//   // Back substitution
//   for (int i = n - 1; i >= 0; i--) {
//     double sum = 0;
//     for (int j = i + 1; j < n; j++) {
//       sum += A_temp->data[i][j] * x->data[j][0];
//     }
//     x->data[i][0] = (b_temp->data[i][0] - sum) / A_temp->data[i][i];
//   }
//
//   matrix_free(A_temp);
//   matrix_free(b_temp);
//
//   return x;
// }
//
// // TODO: Fix
// Matrix *matrix_solve_lu(Matrix *A, Matrix *b) {
//   int n = A->cols;
//
//   // check if the dimensions of the input matrices are compatible
//   if (A->cols != b->rows) {
//     fprintf(stderr, "matrix_solve_lu: incompatible dimensions");
//     return NULL;
//   }
//
//   Matrix *x = matrix_new(n, 1);
//
//   // Copy A and b to temporary matrices
//   Matrix *A_temp = matrix_copy(A);
//   if (A_temp == NULL) {
//     fprintf(stderr, "matrix_solve_lu: failed to copy A");
//     return NULL;
//   }
//   Matrix *b_temp = matrix_copy(b);
//   if (b_temp == NULL) {
//     fprintf(stderr, "matrix_solve_lu: failed to copy b");
//     return NULL;
//   }
//
//   // LU decomposition
//   for (int i = 0; i < n - 1; i++) {
//     // Eliminate the lower rows
//     for (int j = i + 1; j < n; j++) {
//       double c = A_temp->data[j][i] / A_temp->data[i][i];
//       for (int k = i; k < n; k++) {
//         A_temp->data[j][k] -= A_temp->data[i][k] * c;
//       }
//       b_temp->data[j][0] -= b_temp->data[i][0] * c;
//     }
//   }
//
//   // Back substitution
//   for (int i = n - 1; i >= 0; i--) {
//     double sum = 0;
//     for (int j = i + 1; j < n; j++) {
//       sum += A_temp->data[i][j] * x->data[j][0];
//     }
//     x->data[i][0] = (b_temp->data[i][0] - sum) / A_temp->data[i][i];
//   }
//
//   matrix_free(A_temp);
//   matrix_free(b_temp);
//
//   return x;
// }

// Tensor functions
// -----------------------------------------------------------------------------
// Tensor *tensor_new(int rows, int cols, int rank) {
//   Tensor *t = (Tensor *)malloc(sizeof(Tensor));
//
//   if (t == NULL) {
//     fprintf(stderr, "tensor_new: failed to allocate memory");
//     return NULL;
//   }
//
//   t->rows = rows;
//   t->cols = cols;
//   t->rank = rank;
//
//   t->data = (double ***)malloc(rank * sizeof(double **));
//   if (t->data == NULL) {
//     fprintf(stderr, "tensor_new: failed to allocate memory");
//     return NULL;
//   }
//
//   for (int i = 0; i < rank; i++) {
//     t->data[i] = (double **)malloc(rows * sizeof(double *));
//   }
//
//   for (int i = 0; i < rank; i++) {
//     for (int j = 0; j < rows; j++) {
//       t->data[i][j] = (double *)malloc(cols * sizeof(double));
//     }
//   }
//
//   return t;
// }

// void tensor_insert(Tensor *t, Matrix *m, int index) {
//   if (index >= t->rank) {
//     fprintf(stderr, "tensor_insert: index out of bounds");
//     return;
//   }
//
//   t->data[index].rows = m->rows;
//   t->data[index].cols = m->cols;
//   t->data[index] = *m;
// }

// Tensor *tensor_copy(Tensor *src) {
//   Tensor *dst = tensor_new(src->rows, src->cols, src->rank);
//
//   for (int i = 0; i < src->rank; i++) {
//     Matrix *temp = matrix_copy(&(src->data[i]));
//     if (temp == NULL) {
//       fprintf(stderr, "tensor_copy: failed to allocate memory");
//       return NULL;
//     }
//     dst->data[i] = *temp;
//   }
//
//   return dst;
// }
//
// void tensor_print(Tensor *t) {
//   for (int i = 0; i < t->rank; i++) {
//     printf("Tensor rank %d:\n", i);
//     if (t->data[i].rows == 0 || t->data[i].cols == 0) {
//       printf("Empty matrix\n");
//     } else {
//       matrix_print(&(t->data[i]));
//     }
//   }
// }
//
// Tensor *tensor_add(Tensor *t1, Tensor *t2) {
//   if (t1->rank != t2->rank) {
//     fprintf(stderr, "tensor_add: tensors must have the same rank");
//     return NULL;
//   }
//
//   Tensor *result = tensor_new(t1->rank, t1->rows, t1->cols);
//   if (result == NULL) {
//     fprintf(stderr, "tensor_add: failed to allocate memory");
//     return NULL;
//   }
//
//   for (int i = 0; i < t1->rank; i++) {
//     Matrix *temp;
//     temp = matrix_add(&(t1->data[i]), &(t2->data[i]));
//     if (temp == NULL) {
//       fprintf(stderr, "tensor_add: failed to allocate memory");
//       return NULL;
//     }
//     tensor_insert(result, temp, i);
//   }
//
//   return result;
// }
//
// Tensor *tensor_sub(Tensor *t1, Tensor *t2) {
//   if (t1->rank != t2->rank) {
//     fprintf(stderr, "tensor_sub: tensors must have the same rank");
//     return NULL;
//   }
//
//   Tensor *result = tensor_new(t1->rank, t1->rows, t1->cols);
//   if (result == NULL) {
//     fprintf(stderr, "tensor_sub: failed to allocate memory");
//     return NULL;
//   }
//
//   for (int i = 0; i < t1->rank; i++) {
//     Matrix *temp;
//     temp = matrix_sub(&(t1->data[i]), &(t2->data[i]));
//     if (temp == NULL) {
//       fprintf(stderr, "tensor_sub: failed to allocate memory");
//       return NULL;
//     }
//     tensor_insert(result, temp, i);
//   }
//
//   return result;
// }
//
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
