#include "../src/linear_algebra.h"
#include "../src/linear_algebra.c"

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

void test_vector_copy() {
  Vector *v = vector_new(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;
  Vector *copy = vector_copy(v);
  assert(copy != NULL);
  assert(copy->size == 3);
  assert(copy->data[0] == 1);
  assert(copy->data[1] == 2);
  assert(copy->data[2] == 3);
  vector_free(v);
  vector_free(copy);
}

void test_vector_add() {
  Vector *v1 = vector_new(3);
  v1->data[0] = 1;
  v1->data[1] = 2;
  v1->data[2] = 3;
  Vector *v2 = vector_new(3);
  v2->data[0] = 4;
  v2->data[1] = 5;
  v2->data[2] = 6;
  Vector *v3 = vector_add(v1, v2);
  assert(v3 != NULL);
  assert(v3->size == 3);
  assert(v3->data[0] == 5);
  assert(v3->data[1] == 7);
  assert(v3->data[2] == 9);
  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_subtract() {
  Vector *v1 = vector_new(3);
  v1->data[0] = 1;
  v1->data[1] = 2;
  v1->data[2] = 3;
  Vector *v2 = vector_new(3);
  v2->data[0] = 4;
  v2->data[1] = 5;
  v2->data[2] = 6;
  Vector *v3 = vector_sub(v1, v2);
  assert(v3 != NULL);
  assert(v3->size == 3);
  assert(v3->data[0] == -3);
  assert(v3->data[1] == -3);
  assert(v3->data[2] == -3);
  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_scale() {
  Vector *v = vector_new(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;
  Vector *v2 = vector_scale(v, 2);
  assert(v2 != NULL);
  assert(v2->size == 3);
  assert(v2->data[0] == 2);
  assert(v2->data[1] == 4);
  assert(v2->data[2] == 6);
  vector_free(v);
  vector_free(v2);
}

void test_vector_dot() {
  Vector *v1 = vector_new(3);
  v1->data[0] = 1;
  v1->data[1] = 2;
  v1->data[2] = 3;
  Vector *v2 = vector_new(3);
  v2->data[0] = 4;
  v2->data[1] = 5;
  v2->data[2] = 6;
  double dot = vector_dot(v1, v2);
  assert(dot == 32);
  vector_free(v1);
  vector_free(v2);
}

void test_vector_norm() {
  Vector *v = vector_new(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;
  double norm = vector_norm(v);
  assert(norm == sqrt(14));
  vector_free(v);
}

void test_vector_normalize() {
  Vector *v = vector_new(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;
  Vector *v2 = vector_normalize(v);
  assert(v2 != NULL);
  assert(v2->size == 3);
  assert(v2->data[0] == 1 / sqrt(14));
  assert(v2->data[1] == 2 / sqrt(14));
  assert(v2->data[2] == 3 / sqrt(14));
  vector_free(v);
  vector_free(v2);
}

void test_vector_cross() {
  Vector *v1 = vector_new(3);
  v1->data[0] = 1;
  v1->data[1] = 2;
  v1->data[2] = 3;
  Vector *v2 = vector_new(3);
  v2->data[0] = 4;
  v2->data[1] = 5;
  v2->data[2] = 6;
  Vector *v3 = vector_cross(v1, v2);
  assert(v3 != NULL);
  assert(v3->size == 3);
  assert(v3->data[0] == -3);
  assert(v3->data[1] == 6);
  assert(v3->data[2] == -3);
  vector_free(v1);
  vector_free(v2);
  vector_free(v3);
}

void test_vector_from_array() {
  double data[] = {1, 2, 3};
  Vector *v = vector_from_array(3, data);
  assert(v != NULL);
  assert(v->size == 3);
  assert(v->data[0] == 1);
  assert(v->data[1] == 2);
  assert(v->data[2] == 3);
  vector_free(v);
}

void test_vector_to_array() {
  Vector *v = vector_new(3);
  v->data[0] = 1;
  v->data[1] = 2;
  v->data[2] = 3;
  double *data = vector_to_array(v);
  assert(data[0] == 1);
  assert(data[1] == 2);
  assert(data[2] == 3);
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
  assert(m->data[0][0] == 0);
  assert(m->data[0][1] == 0);
  assert(m->data[0][2] == 0);
  assert(m->data[1][0] == 0);
  assert(m->data[1][1] == 0);
  assert(m->data[1][2] == 0);
  matrix_free(m);
}

void test_matrix_free() {
  Matrix *m = matrix_new(2, 3);
  matrix_free(m);
}

void test_matrix_copy() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1;
  m->data[0][1] = 2;
  m->data[0][2] = 3;
  m->data[1][0] = 4;
  m->data[1][1] = 5;
  m->data[1][2] = 6;
  Matrix *m2 = matrix_copy(m);
  assert(m2 != NULL);
  assert(m2->rows == 2);
  assert(m2->cols == 3);
  assert(m2->data[0][0] == 1);
  assert(m2->data[0][1] == 2);
  assert(m2->data[0][2] == 3);
  assert(m2->data[1][0] == 4);
  assert(m2->data[1][1] == 5);
  assert(m2->data[1][2] == 6);
  matrix_free(m);
  matrix_free(m2);
}

void test_matrix_add() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1;
  m->data[0][1] = 2;
  m->data[0][2] = 3;
  m->data[1][0] = 4;
  m->data[1][1] = 5;
  m->data[1][2] = 6;
  Matrix *m2 = matrix_new(2, 3);
  m2->data[0][0] = 1;
  m2->data[0][1] = 2;
  m2->data[0][2] = 3;
  m2->data[1][0] = 4;
  m2->data[1][1] = 5;
  m2->data[1][2] = 6;
  Matrix *m3 = matrix_add(m, m2);
  assert(m3 != NULL);
  assert(m3->rows == 2);
  assert(m3->cols == 3);
  assert(m3->data[0][0] == 2);
  assert(m3->data[0][1] == 4);
  assert(m3->data[0][2] == 6);
  assert(m3->data[1][0] == 8);
  assert(m3->data[1][1] == 10);
  assert(m3->data[1][2] == 12);
  matrix_free(m);
  matrix_free(m2);
  matrix_free(m3);
}

void test_matrix_sub() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1;
  m->data[0][1] = 2;
  m->data[0][2] = 3;
  m->data[1][0] = 4;
  m->data[1][1] = 5;
  m->data[1][2] = 6;
  Matrix *m2 = matrix_new(2, 3);
  m2->data[0][0] = 1;
  m2->data[0][1] = 2;
  m2->data[0][2] = 3;
  m2->data[1][0] = 4;
  m2->data[1][1] = 5;
  m2->data[1][2] = 6;
  Matrix *m3 = matrix_sub(m, m2);
  assert(m3 != NULL);
  assert(m3->rows == 2);
  assert(m3->cols == 3);
  assert(m3->data[0][0] == 0);
  assert(m3->data[0][1] == 0);
  assert(m3->data[0][2] == 0);
  assert(m3->data[1][0] == 0);
  assert(m3->data[1][1] == 0);
  assert(m3->data[1][2] == 0);
  matrix_free(m);
  matrix_free(m2);
  matrix_free(m3);
}

void test_matrix_scale() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1;
  m->data[0][1] = 2;
  m->data[0][2] = 3;
  m->data[1][0] = 4;
  m->data[1][1] = 5;
  m->data[1][2] = 6;
  Matrix *m2 = matrix_scale(m, 2);
  assert(m2 != NULL);
  assert(m2->rows == 2);
  assert(m2->cols == 3);
  assert(m2->data[0][0] == 2);
  assert(m2->data[0][1] == 4);
  assert(m2->data[0][2] == 6);
  assert(m2->data[1][0] == 8);
  assert(m2->data[1][1] == 10);
  assert(m2->data[1][2] == 12);
  matrix_free(m);
  matrix_free(m2);
}

void test_matrix_multiply() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1;
  m->data[0][1] = 2;
  m->data[0][2] = 3;
  m->data[1][0] = 4;
  m->data[1][1] = 5;
  m->data[1][2] = 6;
  Matrix *m2 = matrix_new(3, 2);
  m2->data[0][0] = 1;
  m2->data[0][1] = 2;
  m2->data[1][0] = 3;
  m2->data[1][1] = 4;
  m2->data[2][0] = 5;
  m2->data[2][1] = 6;
  Matrix *m3 = matrix_multiply(m, m2);
  assert(m3 != NULL);
  assert(m3->rows == 2);
  assert(m3->cols == 2);
  assert(m3->data[0][0] == 22);
  assert(m3->data[0][1] == 28);
  assert(m3->data[1][0] == 49);
  assert(m3->data[1][1] == 64);
  matrix_free(m);
  matrix_free(m2);
  matrix_free(m3);
}

// void test_multiply_vector() {
//   Matrix *m = matrix_new(2, 3);
//   m->data[0][0] = 1;
//   m->data[0][1] = 2;
//   m->data[0][2] = 3;
//   m->data[1][0] = 4;
//   m->data[1][1] = 5;
//   m->data[1][2] = 6;
//   Vector *v = vector_new(3);
//   v->data[0] = 1;
//   v->data[1] = 2;
//   v->data[2] = 3;
//   Vector *v2 = matrix_multiply_vector(m, v);
//   assert(v2 != NULL);
//   assert(v2->size == 2);
//   assert(v2->data[0] == 14);
//   assert(v2->data[1] == 32);
//   matrix_free(m);
//   vector_free(v);
//   vector_free(v2);
// }

void test_matrix_transpose() {
  Matrix *m = matrix_new(2, 3);
  m->data[0][0] = 1;
  m->data[0][1] = 2;
  m->data[0][2] = 3;
  m->data[1][0] = 4;
  m->data[1][1] = 5;
  m->data[1][2] = 6;
  Matrix *m2 = matrix_transpose(m);
  assert(m2 != NULL);
  assert(m2->rows == 3);
  assert(m2->cols == 2);
  assert(m2->data[0][0] == 1);
  assert(m2->data[0][1] == 4);
  assert(m2->data[1][0] == 2);
  assert(m2->data[1][1] == 5);
  assert(m2->data[2][0] == 3);
  assert(m2->data[2][1] == 6);
  matrix_free(m);
  matrix_free(m2);
}

void test_matrix_fill() {
  Matrix *m = matrix_new(2, 3);
  matrix_fill(m, 1);
  assert(m->data[0][0] == 1);
  assert(m->data[0][1] == 1);
  assert(m->data[0][2] == 1);
  assert(m->data[1][0] == 1);
  assert(m->data[1][1] == 1);
  assert(m->data[1][2] == 1);
  matrix_free(m);
}

void test_matrix_set() {
  Matrix *m = matrix_new(2, 3);
  double data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  int size = 2 * 3;
  matrix_set(m, data, size);
  assert(m->data[0][0] == 1);
  assert(m->data[0][1] == 2);
  assert(m->data[0][2] == 3);
  assert(m->data[1][0] == 4);
  assert(m->data[1][1] == 5);
  assert(m->data[1][2] == 6);
  matrix_free(m);
}

int main(int argc, char* argv) {
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

  printf("All tests passed\n\n");

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
  // test_matrix_multiply_vector();
  // printf("test_matrix_multiply_vector passed\n");
  test_matrix_transpose();
  printf("test_matrix_transpose passed\n");
  test_matrix_fill();
  printf("test_matrix_fill passed\n");
  test_matrix_set();
  printf("test_matrix_set passed\n");
}
