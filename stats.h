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
  uint32_t N;
  uint32_t K;
  uint32_t n;

} hypergeometric_t;

typedef struct {
  uint32_t r;
  float p;

} negative_binomial_t;

typedef struct {
  uint32_t lambda;

} poisson_t;

typedef struct {
  uint32_t a, b;
  float p;

} uniform_t;

float binomial_mean(binomial_t *b);
float binomial_variance(binomial_t *b);
float binomial_skewness(binomial_t *b);
float binomial_stddev(binomial_t *b);
float binomial_median(binomial_t *b);
float binomial_pmf(binomial_t *b, uint32_t k);
float binomial_cdf(binomial_t *b, uint32_t k);

float bernoulli_mean(bernoulli_t *b);
float bernoulli_variance(bernoulli_t *b);
float bernoulli_stddev(bernoulli_t *b);
float bernoulli_skewness(bernoulli_t *b);
float bernoulli_median(bernoulli_t *b);
float bernoulli_pmf(bernoulli_t *b, uint32_t k);
float bernoulli_cdf(bernoulli_t *b, uint32_t k);

float discrete_uniform_mean(discrete_uniform_t *d);
float discrete_uniform_variance(discrete_uniform_t *d);
float discrete_uniform_stddev(discrete_uniform_t *d);
float discrete_uniform_skewness(discrete_uniform_t *d);
float discrete_uniform_median(discrete_uniform_t *d);
float discrete_uniform_pmf(discrete_uniform_t *d, uint32_t k);
float discrete_uniform_cdf(discrete_uniform_t *d, uint32_t k);

float geometric_mean(geometric_t *g);
float geometric_variance(geometric_t *g);
float geometric_stddev(geometric_t *g);
float geometric_skewness(geometric_t *g);
float geometric_median(geometric_t *g);
float geometric_pmf(geometric_t *g, uint32_t k);
float geometric_cdf(geometric_t *g, uint32_t k);

float hypergeometric_mean(hypergeometric_t *h);
float hypergeometric_variance(hypergeometric_t *h);
float hypergeometric_stddev(hypergeometric_t *h);
float hypergeometric_skewness(hypergeometric_t *h);
float hypergeometric_median(hypergeometric_t *h);
float hypergeometric_pmf(hypergeometric_t *h, uint32_t k);
float hypergeometric_cdf(hypergeometric_t *h, uint32_t k);

float negativebinomal_mean(negative_binomial_t *n);
float negativebinomal_variance(negative_binomial_t *n);
float negativebinomal_stddev(negative_binomial_t *n);
float negativebinomal_skewness(negative_binomial_t *n);
float negativebinomal_median(negative_binomial_t *n);
float negativebinomal_pmf(negative_binomial_t *n, uint32_t k);
float negativebinomal_cdf(negative_binomial_t *n, uint32_t k);

float poisson_mean(poisson_t *p);
float poisson_variance(poisson_t *p);
float poisson_stddev(poisson_t *p);
float poisson_skewness(poisson_t *p);
float poisson_median(poisson_t *p);
float poisson_pmf(poisson_t *p, uint32_t k);
float poisson_cdf(poisson_t *p, uint32_t k);

#endif
