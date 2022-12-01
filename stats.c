#include "stats.h"
#include <stdio.h>
#include <stdlib.h>

float bernoulli_mean(bernoulli_t* ber) {
  return ber->p;
}

float binomial_t_mean(binomial_t* bin) {
  float res = bin->n * bin->p;
}
