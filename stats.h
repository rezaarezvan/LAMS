#ifndef STATS_H
#define STATS_H

#include <stdint.h>

typedef struct {
  uint32_t n;
  float p;

} binomial_t;

typedef struct {
  float p;

} bernoulli_t;

typedef struct {
  uint32_t a;
  uint32_t b;
  float p;

} discrete_uniform_t;

typedef struct {
  float p;

} geometric_t;

typedef struct {
  uint32_t ns;
  uint32_t nf;
  uint32_t n;

} hypergeometric_t;

typedef struct {
  uint32_t r;
  float p;

} negativebinomal_t;

typedef struct {
  uint32_t lambda;

} poisson_t;

#endif
