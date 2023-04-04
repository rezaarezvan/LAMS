#include "linear_algebra.h"

// Vector functions
// -----------------------------------------------------------------------------
Vector *vector_new(size_t n) {
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

Vector *vector_copy(const Vector *v) {
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

Vector *vector_add(const Vector *a, const Vector *b) {
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

Vector *vector_sub(const Vector *a, const Vector *b) {
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

Vector *vector_scale(const Vector *v, double c) {
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

double vector_dot(const Vector *a, const Vector *b) {
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

double vector_norm(const Vector *v) {
  double result = 0;

  for (int i = 0; i < v->size; i++) {
    result += v->data[i] * v->data[i];
  }

  return sqrt(result);
}

Vector *vector_normalize(const Vector *v) {
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

Vector *vector_cross(const Vector *a, const Vector *b) {
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

Vector *vector_from_array(size_t size, const double *array) {
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
double *vector_to_array(const Vector *v) {
  double *result = malloc(sizeof(double) * v->size);

  for (int i = 0; i < v->size; i++) {
    result[i] = v->data[i];
  }

  return result;
}

void vector_print(const Vector *v) {
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
Matrix *matrix_new(size_t rows, size_t cols) {
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
Matrix *matrix_copy(const Matrix *m) {
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

Matrix *matrix_add(const Matrix *a, const Matrix *b) {
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

Matrix *matrix_sub(const Matrix *a, const Matrix *b) {
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

Matrix *matrix_scale(const Matrix *m, double s) {
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

Matrix *matrix_multiply(const Matrix *a, const Matrix *b) {
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

Matrix *matrix_multiply_vector(const Matrix *m, const Vector *v) {
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

Matrix *matrix_transpose(const Matrix *m) {
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

void matrix_set(Matrix *m, const Vector *v, size_t row) {
  for (int i = 0; i < m->cols; i++) {
    m->data[row]->data[i] = v->data[i];
  }
}

void matrix_print(const Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%f ", m->data[i]->data[j]);
    }
    printf("\n");
  }
  printf("\n");
}

Matrix *matrix_identity(size_t size) {
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

// Tensor functions
// -----------------------------------------------------------------------------
Tensor *tensor_new(size_t rows, size_t cols, size_t rank) {
  Tensor *t = malloc(sizeof(Tensor));
  t->rows = rows;
  t->cols = cols;
  t->rank = rank;
  t->data = malloc(rank * sizeof(Matrix));

  for (int i = 0; i < rank; i++) {
    t->data[i] = matrix_new(rows, cols);
  }

  return t;
}

void tensor_free(Tensor *t) {
  if (t == NULL) {
    return;
  }

  for (int i = 0; i < t->rank; i++) {
    matrix_free(t->data[i]);
  }

  free(t->data);
  free(t);
}

void tensor_insert(Tensor *t, const Matrix *m, size_t index) {
  if (index >= t->rank) {
    fprintf(stderr, "tensor_insert: index out of bounds");
    return;
  }

  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      t->data[index]->data[i]->data[j] = m->data[i]->data[j];
    }
  }
}

Tensor *tensor_copy(const Tensor *src) {
  Tensor *result = tensor_new(src->rows, src->cols, src->rank);

  for (int i = 0; i < src->rank; i++) {
    for (int j = 0; j < src->rows; j++) {
      for (int k = 0; k < src->cols; k++) {
        result->data[i]->data[j]->data[k] = src->data[i]->data[j]->data[k];
      }
    }
  }

  return result;
}

void tensor_print(const Tensor *t) {
  for (int i = 0; i < t->rank; i++) {
    printf("Matrix %d\n", i);
    matrix_print(t->data[i]);
  }
}

Tensor *tensor_add(const Tensor *t1, const Tensor *t2) {
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
    Matrix *temp = matrix_add(t1->data[i], t2->data[i]);
    if (temp == NULL) {
      fprintf(stderr, "tensor_add: failed to allocate memory");
      return NULL;
    }

    tensor_insert(result, temp, i);
    matrix_free(temp);
  }

  return result;
}

Tensor *tensor_sub(const Tensor *t1, const Tensor *t2) {
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
    Matrix *temp = matrix_sub(t1->data[i], t2->data[i]);
    if (temp == NULL) {
      fprintf(stderr, "tensor_sub: failed to allocate memory");
      return NULL;
    }
    tensor_insert(result, temp, i);
    matrix_free(temp);
  }

  return result;
}
