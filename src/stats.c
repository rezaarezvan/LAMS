#include "stats.h"
#include <math.h>

// Means

float binomial_mean(binomial_t *bin) { return bin->n * bin->p; }

float bernoulli_mean(bernoulli_t *ber) { return ber->p; }

float discrete_uniform_mean(discrete_uniform_t *uni) {
  return (uni->a + uni->b) / 2;
}

float geometric_mean(geometric_t *geo) { return 1 / geo->p; }

float hypergeometric_mean(hypergeometric_t *hyp) {
  return hyp->n * (hyp->K / hyp->N);
}

float negative_binomial_mean(negative_binomial_t *neg) {
  return neg->r * (1 - neg->p) / neg->p;
}

float poisson_mean(poisson_t *poi) { return poi->lambda; }

float continuous_uniform_mean(continuous_uniform_t *uni) {
  return (uni->a + uni->b) / 2;
}

float normal_mean(normal_t *nor) { return nor->mu; }

// Variances

float binomial_variance(binomial_t *bin) {
  return bin->n * bin->p * (1 - bin->p);
}

float bernoulli_variance(bernoulli_t *ber) { return ber->p * (1 - ber->p); }

float discrete_uniform_variance(discrete_uniform_t *uni) {
  return (uni->b - uni->a) * (uni->b - uni->a) / 12;
}

float geometric_variance(geometric_t *geo) {
  return (1 - geo->p) / (geo->p * geo->p);
}

float hypergeometric_variance(hypergeometric_t *hyp) {
  return hyp->n * (hyp->K / hyp->N) * (1 - (hyp->K / hyp->N)) *
         ((hyp->N - hyp->n) / (hyp->N - 1));
}

float negative_binomial_variance(negative_binomial_t *neg) {
  return neg->r * (1 - neg->p) / (neg->p * neg->p);
}

float poisson_variance(poisson_t *poi) { return poi->lambda; }

float continuous_uniform_variance(continuous_uniform_t *uni) {
  return ((uni->b - uni->a) * (uni->b - uni->a)) / 12;
}

float normal_variance(normal_t *nor) { return nor->sigma * nor->sigma; }

// Standard deviations

float binomial_std_dev(binomial_t *bin) { return sqrt(binomial_variance(bin)); }

float bernoulli_std_dev(bernoulli_t *ber) {
  return sqrt(bernoulli_variance(ber));
}

float discrete_uniform_std_dev(discrete_uniform_t *uni) {
  return sqrt(discrete_uniform_variance(uni));
}

float hypergeometric_std_dev(hypergeometric_t *hyp) {
  return sqrt(hypergeometric_variance(hyp));
}

float geometric_std_dev(geometric_t *geo) {
  return sqrt(geometric_variance(geo));
}

float negative_binomial_std_dev(negative_binomial_t *neg) {
  return sqrt(negative_binomial_variance(neg));
}

float poisson_std_dev(poisson_t *poi) { return sqrt(poisson_variance(poi)); }

float continuous_uniform_std_dev(continuous_uniform_t *uni) {
  return sqrt(continuous_uniform_variance(uni));
}

float normal_std_dev(normal_t *nor) { return sqrt(normal_variance(nor)); }

// Skewness

float binomial_skewness(binomial_t *bin) {
  return (1 - 2 * bin->p) / binomial_std_dev(bin);
}

float bernoulli_skewness(bernoulli_t *ber) {
  return (1 - 2 * ber->p) / bernoulli_std_dev(ber);
}

float discrete_uniform_skewness(discrete_uniform_t *uni) { return 0; }

float geometric_skewness(geometric_t *geo) {
  return (1 - 2 * geo->p) / geometric_std_dev(geo);
}

float hypergeometric_skewness(hypergeometric_t *hyp) {
  return (hyp->N - 2 * hyp->K) * sqrt(hyp->N - 1) /
         (hyp->n * (hyp->N - hyp->n) * hyp->K * (hyp->N - hyp->K));
}

float negative_binomial_skewness(negative_binomial_t *neg) {
  return (1 - 2 * neg->p) / negative_binomial_std_dev(neg);
}

float poisson_skewness(poisson_t *poi) { return 1 / poi->lambda; }

float continuous_uniform_skewness(continuous_uniform_t *uni) { return 0; }

float normal_skewness(normal_t *nor) { return 0; }

// Median

float binomial_median(binomial_t *bin) { return floor(bin->n * bin->p + 0.5); }

float bernoulli_median(bernoulli_t *ber) { return floor(ber->p + 0.5); }

float discrete_uniform_median(discrete_uniform_t *dis) {
  return floor((dis->a + dis->b) / 2 + 0.5);
}

float geometric_median(geometric_t *geo) { return floor(1 / geo->p + 0.5); }

float negative_binomial_median(negative_binomial_t *neg) {
  return floor(neg->r * neg->p + 0.5);
}

float hypergeometric_median(hypergeometric_t *hyp) {
  return floor((hyp->n + 1) * (hyp->K + 1) / (hyp->N + 2) + 0.5);
}

float poisson_median(poisson_t *poi) { return floor(poi->lambda + 0.5); }

float continuous_uniform_median(continuous_uniform_t *uni) {
  return (uni->a + uni->b) / 2;
}

float normal_median(normal_t *nor) { return nor->mu; }

// PMF

float binomial_coefficient(int n, int k) {
  float result = 1;
  for (int i = 1; i <= k; i++) {
    result *= (n - k + i) / (float)i;
  }
  return result;
}

float binomial_pmf(binomial_t *bin, uint32_t k) {
  return binomial_coefficient(bin->n, k) * pow(bin->p, k) *
         pow(1 - bin->p, bin->n - k);
}

float bernoulli_pmf(bernoulli_t *ber, uint32_t k) {
  return k == 0 ? 1 - ber->p : ber->p;
}

float discrete_uniform_pmf(discrete_uniform_t *dis, uint32_t k) {
  return k >= dis->a && k <= dis->b ? 1 / (float)(dis->b - dis->a + 1) : 0;
}

float geometric_pmf(geometric_t *geo, uint32_t k) {
  return pow(1 - geo->p, k) * geo->p;
}

float hypergeometric_pmf(hypergeometric_t *hyp, uint32_t k) {
  return binomial_coefficient(hyp->K, k) *
         binomial_coefficient(hyp->N - hyp->K, hyp->n - k) /
         binomial_coefficient(hyp->N, hyp->n);
}

float negative_binomial_pmf(negative_binomial_t *neg, uint32_t k) {
  return binomial_coefficient(k + neg->r - 1, k) * pow(neg->p, neg->r) *
         pow(1 - neg->p, k);
}

float factorial(uint32_t n) {
  float result = 1;
  for (int i = 1; i <= n; i++) {
    result *= i;
  }
  return result;
}

float poisson_pmf(poisson_t *poi, uint32_t k) {
  return pow(poi->lambda, k) * exp(-poi->lambda) / factorial(k);
}

float continuous_uniform_pmf(continuous_uniform_t *uni, float x) {
  return x >= uni->a && x <= uni->b ? 1 / (uni->b - uni->a) : 0;
}

float normal_pmf(normal_t *nor, float x) {
  return exp(-pow(x - nor->mu, 2) / (2 * pow(nor->sigma, 2))) /
         (sqrt(2 * M_PI) * nor->sigma);
}

// CDF

float binomial_cdf(binomial_t *bin, uint32_t k) {
  float result = 0;
  for (int i = 0; i <= k; i++) {
    result += binomial_pmf(bin, i);
  }
  return result;
}

float bernoulli_cdf(bernoulli_t *ber, uint32_t k) {
  return k == 0 ? 1 - ber->p : 1;
}

float discrete_uniform_cdf(discrete_uniform_t *dis, uint32_t k) {
  return k >= dis->b   ? 1
         : k >= dis->a ? (k - dis->a + 1) / (float)(dis->b - dis->a + 1)
                       : 0;
}

float geometric_cdf(geometric_t *geo, uint32_t k) {
  return 1 - pow(1 - geo->p, k + 1);
}

float hypergeometric_cdf(hypergeometric_t *hyp, uint32_t k) {
  float result = 0;
  for (int i = 0; i <= k; i++) {
    result += hypergeometric_pmf(hyp, i);
  }
  return result;
}

float negative_binomial_cdf(negative_binomial_t *neg, uint32_t k) {
  float result = 0;
  for (int i = 0; i <= k; i++) {
    result += negative_binomial_pmf(neg, i);
  }
  return result;
}

float poisson_cdf(poisson_t *poi, uint32_t k) {
  float result = 0;
  for (int i = 0; i <= k; i++) {
    result += poisson_pmf(poi, i);
  }
  return result;
}

float continuous_uniform_cdf(continuous_uniform_t *uni, float x) {
  return x >= uni->b ? 1 : x >= uni->a ? (x - uni->a) / (uni->b - uni->a) : 0;
}

float normal_cdf(normal_t *nor, float x) {
  return 0.5 * (1 + erf((x - nor->mu) / (nor->sigma * sqrt(2))));
}
