#ifndef GLMRGAME_NM_H_
#define GLMRGAME_NM_H_


#define TOL 1e-4

template <typename REAL>
static inline void set_nm_opts(const int maxiter, optimset_t<REAL> *const opts)
{
  opts->tolx = (REAL) TOL;  // tolerance on the simplex solutions coordinates
  opts->tolf = (REAL) TOL;  // tolerance on the function value
  opts->max_iter = maxiter; // maximum number of allowed iterations
  opts->max_eval = maxiter; // maximum number of allowed function evaluations
  opts->verbose = 0;        // toggle verbose output during minimization
}


#endif
