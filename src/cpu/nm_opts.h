#ifndef GLMRGAME_NM_H_
#define GLMRGAME_NM_H_

#include "../nelder-mead/nelder_mead.h"

// tolx - tolerance on the simplex solutions coordinates
// tolf - tolerance on the function value
// max_iter - maximum number of allowed iterations
// max_eval - maximum number of allowed function evaluations
// verbose - toggle verbose output during minimization

static inline void set_nm_opts(const int maxiter, optimset_t *const opts)
{
  #define TOL 1e-8
  opts->tolx = (double) TOL;
  opts->tolf = (double) TOL;
  opts->max_iter = maxiter;
  opts->max_eval = maxiter;
  opts->verbose = 0;
  #undef TOL
}


#endif
