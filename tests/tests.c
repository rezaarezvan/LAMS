#include "../src/linear_algebra.h"

// Unit tests
// -----------------------------------------------------------------------------
// Vector tests
void test_vector_new() {
  Vector *v = vector_new(3);

  assert(v != NULL);
  assert(v->size == 3);
  assert(v->data != NULL);

  vector_free(v);
}

void test_vector_free() {
  Vector *v = vector_new(3);

  vector_free(v);
  v = NULL;
  assert(v == NULL);
}

void test_vector_from_array() {
  double data[] = {1.0, 2.0, 3.0};
  Vector *v = vector_from_array(3, data);

  assert(v != NULL);
  assert(v->size == 3);

  for (int i = 0; i < v->size; i++) {
    assert(v->data[i] == data[i]);
  }

  vector_free(v);
}

void test_vector_copy() {
  double data[] = {1.0, 2.0, 3.0};
  int size = 3;

  Vector *v = vector_from_array(size, data);
  Vector *copy = vector_copy(v);

  assert(copy != NULL);
  assert(copy->size == size);

  for (int i = 0; i < size; i++) {
    assert(copy->data[i] == v->data[i]);
  }

  vector_free(v);
  vector_free(copy);
}

void test_vector_add() {
  double data1[] = {1.0, 2.0, 3.0};
  double data2[] = {4.0, 5.0, 6.0};
  int size = 3;

  Vector *v1 = vector_from_array(size, data1);
  Vector *v2 = vector_from_array(size, data2);
  Vector *v3 = vector_add(v1, v2);

  assert(v3 != NULL);
  assert(v3->size == 3);
  for (int i = 0; i < size; i++) {
    assert(v3->data[i] == v1->data[i] + v2->data[i]);
  }
  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_subtract() {
  double data1[] = {1.0, 2.0, 3.0};
  double data2[] = {4.0, 5.0, 6.0};
  int size = 3;

  Vector *v1 = vector_from_array(size, data1);
  Vector *v2 = vector_from_array(size, data2);
  Vector *v3 = vector_sub(v1, v2);

  assert(v3 != NULL);
  assert(v3->size == 3);

  for (int i = 0; i < size; i++) {
    assert(v3->data[i] == v1->data[i] - v2->data[i]);
  }

  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_scale() {
  double data[] = {1.0, 2.0, 3.0};
  int size = 3;
  int scale = 2;

  Vector *v = vector_from_array(size, data);
  Vector *v2 = vector_scale(v, scale);

  assert(v2 != NULL);
  assert(v2->size == 3);

  for (int i = 0; i < size; i++) {
    assert(v2->data[i] == v->data[i] * scale);
  }
  vector_free(v);
  vector_free(v2);
}

void test_vector_dot() {
  Vector *v1 = vector_new(3);
  v1->data[0] = 1.0;
  v1->data[1] = 2.0;
  v1->data[2] = 3.0;
  Vector *v2 = vector_new(3);
  v2->data[0] = 4.0;
  v2->data[1] = 5.0;
  v2->data[2] = 6.0;
  double dot = vector_dot(v1, v2);
  assert(dot == 32.0);
  vector_free(v1);
  vector_free(v2);
}

void test_vector_norm() {
  Vector *v = vector_new(3);
  v->data[0] = 1.0;
  v->data[1] = 2.0;
  v->data[2] = 3.0;
  double norm = vector_norm(v);
  assert(norm == sqrt(14.0));
  vector_free(v);
}

void test_vector_normalize() {
  Vector *v = vector_new(3);
  v->data[0] = 1.0;
  v->data[1] = 2.0;
  v->data[2] = 3.0;
  Vector *v2 = vector_normalize(v);
  assert(v2 != NULL);
  assert(v2->size == 3);
  assert(v2->data[0] == 1.0 / sqrt(14.0));
  assert(v2->data[1] == 2.0 / sqrt(14.0));
  assert(v2->data[2] == 3.0 / sqrt(14.0));
  vector_free(v);
  vector_free(v2);
}

void test_vector_cross() {
  Vector *v1 = vector_new(3);
  v1->data[0] = 1.0;
  v1->data[1] = 2.0;
  v1->data[2] = 3.0;
  Vector *v2 = vector_new(3);
  v2->data[0] = 4.0;
  v2->data[1] = 5.0;
  v2->data[2] = 6.0;
  Vector *v3 = vector_cross(v1, v2);
  assert(v3 != NULL);
  assert(v3->size == 3);
  assert(v3->data[0] == -3.0);
  assert(v3->data[1] == 6.0);
  assert(v3->data[2] == -3.0);
  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_to_array() {
  Vector *v = vector_new(3);
  v->data[0] = 1.0;
  v->data[1] = 2.0;
  v->data[2] = 3.0;
  double *data = vector_to_array(v);
  assert(data[0] == 1.0);
  assert(data[1] == 2.0);
  assert(data[2] == 3.0);
  free(data);
  vector_free(v);
}
// Matrix tests
// -----------------------------------------------------------------------------

void test_matrix_new() {
  Matrix *m = matrix_new(2, 3);
  assert(m != NULL);
  assert(m->rows == 2);
  assert(m->cols == 3);
  matrix_free(m);
}

void test_matrix_free() {
  Matrix *m = matrix_new(0, 0);
  matrix_free(m);
}

void test_matrix_copy() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Matrix *m2 = matrix_copy(m);
  assert(m2 != NULL);
  assert(m2->rows == 2);
  assert(m2->cols == 3);
  assert(m2->data[0][0] == 1.0);
  assert(m2->data[0][1] == 2.0);
  assert(m2->data[0][2] == 3.0);
  assert(m2->data[1][0] == 4.0);
  assert(m2->data[1][1] == 5.0);
  assert(m2->data[1][2] == 6.0);
  matrix_free(m);
  matrix_free(m2);
}

void test_matrix_add() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Matrix *m2 = matrix_new(2, 3);
  m2->data[0][0] = 1.0;
  m2->data[0][1] = 2.0;
  m2->data[0][2] = 3.0;
  m2->data[1][0] = 4.0;
  m2->data[1][1] = 5.0;
  m2->data[1][2] = 6.0;
  Matrix *m3 = matrix_add(m, m2);
  assert(m3 != NULL);
  assert(m3->rows == 2);
  assert(m3->cols == 3);
  assert(m3->data[0][0] == 2.0);
  assert(m3->data[0][1] == 4.0);
  assert(m3->data[0][2] == 6.0);
  assert(m3->data[1][0] == 8.0);
  assert(m3->data[1][1] == 10.0);
  assert(m3->data[1][2] == 12.0);
  matrix_free(m);
  matrix_free(m2);
  matrix_free(m3);
}

void test_matrix_sub() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Matrix *m2 = matrix_new(2, 3);
  m2->data[0][0] = 1.0;
  m2->data[0][1] = 2.0;
  m2->data[0][2] = 3.0;
  m2->data[1][0] = 4.0;
  m2->data[1][1] = 5.0;
  m2->data[1][2] = 6.0;
  Matrix *m3 = matrix_sub(m, m2);
  assert(m3 != NULL);
  assert(m3->rows == 2);
  assert(m3->cols == 3);
  assert(m3->data[0][0] == 0.0);
  assert(m3->data[0][1] == 0.0);
  assert(m3->data[0][2] == 0.0);
  assert(m3->data[1][0] == 0.0);
  assert(m3->data[1][1] == 0.0);
  assert(m3->data[1][2] == 0.0);
  matrix_free(m);
  matrix_free(m2);
  matrix_free(m3);
}

void test_matrix_scale() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Matrix *m2 = matrix_scale(m, 2);
  assert(m2 != NULL);
  assert(m2->rows == 2);
  assert(m2->cols == 3);
  assert(m2->data[0][0] == 2.0);
  assert(m2->data[0][1] == 4.0);
  assert(m2->data[0][2] == 6.0);
  assert(m2->data[1][0] == 8.0);
  assert(m2->data[1][1] == 10.0);
  assert(m2->data[1][2] == 12.0);
  matrix_free(m);
  matrix_free(m2);
}

void test_matrix_multiply() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Matrix *m2 = matrix_new(3, 2);
  m2->data[0][0] = 1.0;
  m2->data[0][1] = 2.0;
  m2->data[1][0] = 3.0;
  m2->data[1][1] = 4.0;
  m2->data[2][0] = 5.0;
  m2->data[2][1] = 6.0;
  Matrix *m3 = matrix_multiply(m, m2);
  assert(m3 != NULL);
  assert(m3->rows == 2);
  assert(m3->cols == 2);
  assert(m3->data[0][0] == 22.0);
  assert(m3->data[0][1] == 28.0);
  assert(m3->data[1][0] == 49.0);
  assert(m3->data[1][1] == 64.0);
  matrix_free(m);
  matrix_free(m2);
  matrix_free(m3);
}

void test_matrix_multiply_vector() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Vector *v = vector_new(3);
  v->data[0] = 1.0;
  v->data[1] = 2.0;
  v->data[2] = 3.0;

  Matrix *res = matrix_multiply_vector(m, v);
  assert(res != NULL);
  assert(res->rows == 2);
  assert(res->cols == 1);
  assert(res->data[0][0] == 14.0);
  assert(res->data[1][0] == 32.0);

  matrix_free(res);
  matrix_free(m);
  vector_free(v);
}

void test_matrix_transpose() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1.0;
  m->data[0][1] = 2.0;
  m->data[0][2] = 3.0;
  m->data[1][0] = 4.0;
  m->data[1][1] = 5.0;
  m->data[1][2] = 6.0;
  Matrix *m2 = matrix_transpose(m);
  assert(m2 != NULL);
  assert(m2->rows == 3);
  assert(m2->cols == 2);
  assert(m2->data[0][0] == 1.0);
  assert(m2->data[0][1] == 4.0);
  assert(m2->data[1][0] == 2.0);
  assert(m2->data[1][1] == 5.0);
  assert(m2->data[2][0] == 3.0);
  assert(m2->data[2][1] == 6.0);
  matrix_free(m);
  matrix_free(m2);
}

void test_matrix_fill() {
  Matrix *m = matrix_new(2, 3);
  matrix_fill(m, 1);
  assert(m->data[0][0] == 1.0);
  assert(m->data[0][1] == 1.0);
  assert(m->data[0][2] == 1.0);
  assert(m->data[1][0] == 1.0);
  assert(m->data[1][1] == 1.0);
  assert(m->data[1][2] == 1.0);
  matrix_free(m);
}

void test_matrix_set() {
  Matrix *m = matrix_new(2, 3);
  double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  int size = 2 * 3;
  matrix_set(m, data, size);
  assert(m->data[0][0] == 1.0);
  assert(m->data[0][1] == 2.0);
  assert(m->data[0][2] == 3.0);
  assert(m->data[1][0] == 4.0);
  assert(m->data[1][1] == 5.0);
  assert(m->data[1][2] == 6.0);
  matrix_free(m);
}

void test_matrix_identity() {
  Matrix *m = matrix_identity(3);
  assert(m->data[0][0] == 1.0);
  assert(m->data[0][1] == 0.0);
  assert(m->data[0][2] == 0.0);
  assert(m->data[1][0] == 0.0);
  assert(m->data[1][1] == 1.0);
  assert(m->data[1][2] == 0.0);
  assert(m->data[2][0] == 0.0);
  assert(m->data[2][1] == 0.0);
  assert(m->data[2][2] == 1.0);
  matrix_free(m);
}

void test_tensor_new() {
  Tensor *t = tensor_new(1, 1, 1);
  assert(t != NULL);
  assert(t->data != NULL);
  for (int i = 0; i < t->rank; i++) {
    assert(&(t->data[i]) != NULL);
  }
  assert(t->rows == 1);
  assert(t->cols == 1);
  assert(t->rank == 1);
  tensor_free(t);
}

void test_tensor_free() {
  Tensor *t = tensor_new(3, 3, 3);
  tensor_free(t);
}

void test_tensor_insert() {
  Tensor *t = tensor_new(3, 3, 3);
  Matrix *m = matrix_new(3, 3);
  matrix_fill(m, 1);
  tensor_insert(t, m, 0);

  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      assert(t->data[0][i][j] == 1);
    }
  }

  tensor_free(t);
  matrix_free(m);
}

void test_tensor_copy() {
  Tensor *t = tensor_new(3, 3, 3);
  Matrix *m = matrix_new(3, 3);
  matrix_fill(m, 1);
  tensor_insert(t, m, 0);
  Tensor *t2 = tensor_copy(t);

  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      assert(t2->data[0][i][j] == 1);
    }
  }

  tensor_free(t);
  tensor_free(t2);
  matrix_free(m);
}

void test_tensor_add() {
  Tensor *t = tensor_new(3, 3, 3);
  Matrix *m = matrix_new(3, 3);
  matrix_fill(m, 1);
  tensor_insert(t, m, 0);
  Tensor *t2 = tensor_copy(t);
  Tensor *t3 = tensor_add(t, t2);
  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      assert(t3->data[0][i][j] == 2);
    }
  }
  tensor_free(t);
  tensor_free(t2);
  tensor_free(t3);
  matrix_free(m);
}

void test_tensor_sub() {
  Tensor *t = tensor_new(3, 3, 3);
  Matrix *m = matrix_new(3, 3);
  matrix_fill(m, 1);
  tensor_insert(t, m, 0);
  Tensor *t2 = tensor_copy(t);
  Tensor *t3 = tensor_sub(t, t2);
  for (int i = 0; i < t->rows; i++) {
    for (int j = 0; j < t->cols; j++) {
      assert(t3->data[0][i][j] == 0);
    }
  }
  tensor_free(t);
  tensor_free(t2);
  tensor_free(t3);
  matrix_free(m);
}

int main() {
  test_vector_new();
  printf("test_vector_new passed\n");
  test_vector_free();
  printf("test_vector_free passed\n");
  test_vector_copy();
  printf("test_vector_copy passed\n");
  test_vector_add();
  printf("test_vector_add passed\n");
  test_vector_subtract();
  printf("test_vector_subtract passed\n");
  test_vector_scale();
  printf("test_vector_scale passed\n");
  test_vector_dot();
  printf("test_vector_dot passed\n");
  test_vector_norm();
  printf("test_vector_norm passed\n");
  test_vector_normalize();
  printf("test_vector_normalize passed\n");
  test_vector_from_array();
  printf("test_vector_from_array passed\n");
  test_vector_to_array();
  printf("test_vector_to_array passed\n");

  printf("\nAll Vector tests passed\n\n");

  test_matrix_new();
  printf("test_matrix_new passed\n");
  test_matrix_free();
  printf("test_matrix_free passed\n");
  test_matrix_copy();
  printf("test_matrix_copy passed\n");
  test_matrix_add();
  printf("test_matrix_add passed\n");
  test_matrix_sub();
  printf("test_matrix_sub passed\n");
  test_matrix_scale();
  printf("test_matrix_scale passed\n");
  test_matrix_multiply();
  printf("test_matrix_multiply passed\n");
  test_matrix_multiply_vector();
  printf("test_matrix_multiply_vector passed\n");
  test_matrix_transpose();
  printf("test_matrix_transpose passed\n");
  test_matrix_fill();
  printf("test_matrix_fill passed\n");
  test_matrix_set();
  printf("test_matrix_set passed\n");
  test_matrix_identity();
  printf("test_matrix_identity passed\n");

  printf("\nAll Matrix tests passed\n\n");

  test_tensor_new();
  printf("test_tensor_new passed\n");
  test_tensor_free();
  printf("test_tensor_free passed\n");
  test_tensor_insert();
  printf("test_tensor_insert passed\n");
  test_tensor_copy();
  printf("test_tensor_copy passed\n");
  test_tensor_add();
  printf("test_tensor_add passed\n");
  test_tensor_sub();
  printf("test_tensor_sub passed\n");

  printf("\nAll Tensor tests passed\n\n");
}
