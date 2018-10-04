// #include <float/float32.h>
// #include <float/slapack.h>
#include <mpi.h>
#include <Rinternals.h>
#include <stdlib.h>

#include "nelder-mead/nelder_mead.h"

#include "common.h"
#include "mpi_utils.h"
#include "svm_cpu.h"

#define TOL 1e-4


static inline void set_nm_opts(const int maxiter, optimset_t *const restrict opts)
{
  opts->tolx = TOL;         // tolerance on the simplex solutions coordinates
  opts->tolf = TOL;         // tolerance on the function value
  opts->max_iter = maxiter; // maximum number of allowed iterations
  opts->max_eval = maxiter; // maximum number of allowed function evaluations
  opts->verbose = 0;        // toggle verbose output during minimization
}



SEXP R_svm(SEXP x, SEXP y, SEXP maxiter, SEXP comm_)
{
  SEXP ret, ret_names, w, niters;
  optimset_t opts;
  MPI_Comm *comm = get_mpi_comm_from_Robj(comm_);
  const int m = nrows(x);
  const int n = ncols(x);
  
  PROTECT(ret = allocVector(VECSXP, 2));
  PROTECT(ret_names = allocVector(STRSXP, 2));
  PROTECT(w = allocVector(REALSXP, n));
  PROTECT(niters = allocVector(INTSXP, 1));
  
  SET_VECTOR_ELT(ret, 0, w);
  SET_VECTOR_ELT(ret, 1, niters);
  SET_STRING_ELT(ret_names, 0, mkChar("w"));
  SET_STRING_ELT(ret_names, 1, mkChar("niters"));
  setAttrib(ret, R_NamesSymbol, ret_names);
  
  set_nm_opts(INTEGER(maxiter)[0], &opts);
  svm_cpu(m, n, REAL(x), INTEGER(y), REAL(w), comm, &opts);
  
  UNPROTECT(4);
  return ret;
}
