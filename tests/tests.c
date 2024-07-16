#include "../src/linear_algebra.h"
#include "../src/stats.h"

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

  for (uint32_t i = 0; i < v->size; i++) {
    assert(v->data[i] == data[i]);
  }

  vector_free(v);
}

void test_vector_copy() {
  double data[] = {1.0, 2.0, 3.0};
  uint32_t size = 3;

  Vector *v = vector_from_array(size, data);
  Vector *copy = vector_copy(v);

  assert(copy != NULL);
  assert(copy->size == size);

  for (uint32_t i = 0; i < size; i++) {
    assert(copy->data[i] == v->data[i]);
  }

  vector_free(v);
  vector_free(copy);
}

void test_vector_add() {
  double data1[] = {1.0, 2.0, 3.0};
  double data2[] = {4.0, 5.0, 6.0};
  uint32_t size = 3;

  Vector *v1 = vector_from_array(size, data1);
  Vector *v2 = vector_from_array(size, data2);
  Vector *v3 = vector_add(v1, v2);

  assert(v3 != NULL);
  assert(v3->size == 3);
  for (uint32_t i = 0; i < size; i++) {
    assert(v3->data[i] == v1->data[i] + v2->data[i]);
  }
  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_subtract() {
  double data1[] = {1.0, 2.0, 3.0};
  double data2[] = {4.0, 5.0, 6.0};
  uint32_t size = 3;

  Vector *v1 = vector_from_array(size, data1);
  Vector *v2 = vector_from_array(size, data2);
  Vector *v3 = vector_sub(v1, v2);

  assert(v3 != NULL);
  assert(v3->size == 3);

  for (uint32_t i = 0; i < size; i++) {
    assert(v3->data[i] == v1->data[i] - v2->data[i]);
  }

  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_scale() {
  double data[] = {1.0, 2.0, 3.0};
  uint32_t size = 3;
  uint32_t scale = 2;

  Vector *v = vector_from_array(size, data);
  Vector *v2 = vector_scale(v, scale);

  assert(v2 != NULL);
  assert(v2->size == 3);

  for (uint32_t i = 0; i < size; i++) {
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
  assert(m->data != NULL);
  for (uint32_t i = 0; i < m->rows; i++) {
    assert(m->data[i] != NULL);
    assert(m->data[i]->size == m->cols);
  }
  matrix_free(m);
}

void test_matrix_free() {
  Matrix *m = matrix_new(0, 0);
  matrix_free(m);
}

// void test_tensor_new() {
//   Tensor *t = tensor_new(2, 2, 3);
//   assert(t != NULL);
//   assert(t->rank == 2);
//   assert(t->num_matrices == 2);
//   assert(t->data != NULL);
//   for (uint32_t i = 0; i < t->num_matrices; i++) {
//     assert(t->data[i] != NULL);
//     assert(t->data[i]->rows == 2);
//     assert(t->data[i]->cols == 3);
//   }
//   tensor_free(t);
// }
//
// void test_tensor_free() {
//   Tensor *t = tensor_new(2, 2, 3);
//   tensor_free(t);
//   // Note: We can't check if t is NULL after freeing,
//   // as the pointer itself isn't modified.
// }

// Stats tests
// -----------------------------------------------------------------------------

void test_binomial_new() {
  binomial_t b = {10, 0.5};
  assert(b.n == 10);
  assert(b.p == 0.5);
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
  printf("\nAll Matrix tests passed\n\n");

  // test_tensor_new();
  // printf("test_tensor_new passed\n");
  // test_tensor_free();
  // printf("test_tensor_free passed\n");
  // printf("\nAll Tensor tests passed\n\n");

  test_binomial_new();
  printf("test_binomial_new passed\n");

  printf("\nAll Stats tests passed\n\n");
}
