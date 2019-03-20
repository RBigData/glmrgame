// #include <float/float32.h>
// #include <float/slapack.h>
#include <mpi.h>
#include <Rinternals.h>
#include <stdlib.h>
#include <string.h>

#include "../Rmpi.h"

#include "blas.h"
#include "nm_opts.h"


typedef struct {
  int m;
  int n;
  const double *restrict x;
  const int *restrict y;
  double *restrict work;
  MPI_Comm *restrict comm;
} svm_param_t;



static inline double euc_norm_sq(const int n, const double *const restrict x)
{
  double norm = 0.0;
  for (int i=0; i<n; i++)
    norm += x[i]*x[i];
  
  return norm;
}



static inline double hinge_loss_sum(const int n, double *const restrict x)
{
  double s = 0.0;
  for (int i=0; i<n; i++)
  {
    if (x[i] > 0.0)
      s += x[i];
  }
  
  return s;
}



static inline double svm_cost(const int m, const int n, const double *const restrict x,
  const int *const restrict y, const double *const restrict w,
  double *const restrict work, const MPI_Comm *const restrict comm)
{
  int check;
  double J;
  
  // J_local = 1/m * sum(hinge_loss(1.0 - DATA(y)*matmult(DATA(x), w)))
  mvm(m, n, x, w, work);
  for (int i=0; i<m; i++)
    work[i] = 1.0 - y[i]*work[i];
  
  J = ((double) 1.0/m) * hinge_loss_sum(m, work);
  
  // J = allreduce(J_local) + 1/m * 0.5 * norm2(w)
  check = MPI_Allreduce(MPI_IN_PLACE, &J, 1, MPI_DOUBLE, MPI_SUM, *comm);
  MPI_CHECK(comm, check);
  
  J += ((double) 1.0/m) * 0.5 * euc_norm_sq(n, w);
  
  return J;
}



static inline void svm_nmwrap(int n, point_t *point, const void *arg)
{
  const svm_param_t *args = (const svm_param_t*) arg;
  point->fx = svm_cost(args->m, n, args->x, args->y, point->x, args->work, args->comm);
}



static inline void svm(const int m, const int n, const double *const restrict x,
  const int *const restrict y, double *const restrict w, MPI_Comm *const restrict comm,
  optimset_t *const restrict optimset)
{
  svm_param_t args;
  point_t start, solution;
  
  double *work = malloc(m * sizeof(*work));
  
  start.x = w;
  memset(w, 0, n*sizeof(*w));
  
  args.m = m;
  args.n = n;
  args.x = x;
  args.y = y;
  args.work = work;
  args.comm = comm;
  
  nelder_mead(n, &start, &solution, &svm_nmwrap, &args, optimset);
  
  for (int i=0; i<n; i++)
    w[i] = solution.x[i];
  
  free(solution.x);
  free(work);
}



SEXP R_svm(SEXP x, SEXP y, SEXP maxiter, SEXP comm_)
{
  SEXP ret, ret_names, w, niters;
  optimset_t opts;
  MPI_Comm comm = get_mpi_comm_from_Robj(comm_);
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
  svm(m, n, REAL(x), INTEGER(y), REAL(w), &comm, &opts);
  
  UNPROTECT(4);
  return ret;
}
