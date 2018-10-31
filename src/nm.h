#ifndef GLMRGAME_NM_H_
#define GLMRGAME_NM_H_


#ifdef __cplusplus
extern "C" {
#endif
#include "nelder-mead/nelder_mead.h"
#ifdef __cplusplus
}
#endif

#define TOL 1e-4

static inline void set_nm_opts(const int maxiter, optimset_t *const opts)
{
  opts->tolx = TOL;         // tolerance on the simplex solutions coordinates
  opts->tolf = TOL;         // tolerance on the function value
  opts->max_iter = maxiter; // maximum number of allowed iterations
  opts->max_eval = maxiter; // maximum number of allowed function evaluations
  opts->verbose = 0;        // toggle verbose output during minimization
}


#endif
