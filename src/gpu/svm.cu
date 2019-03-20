#include <cstdlib>
#include <float/float32.h>
#include <Rinternals.h>

#include "../mpi_utils.h"
#include "../nelder-mead/nelder_mead.hpp"
#include "../nm.h"

#include "blas.hh"
#include "cu_utils.hh"
#include "mpicxx.hh"
#include "restrict.h"


template <typename REAL>
struct svm_param_t {
  cublasHandle_t handle;
  int m;
  int n;
  const REAL *restrict x;
  const int *restrict y;
  REAL *restrict w;
  REAL *restrict work;
  REAL *restrict s;
  MPI_Comm comm;
};



template <typename REAL>
__global__ static void hinge_loss_sum(REAL *s, const int m, const int *const restrict y, const REAL *const restrict work)
{
  int tid = threadIdx.x;
  int i = tid + blockIdx.x*blockDim.x;
  
  if (i >= m)
    return;
  
  __shared__ REAL temp[TPB];
  
  REAL tmp = 1.0 - y[i]*work[i];
  if (tmp < 0.0)
    temp[tid] = 0.0;  
  else
    temp[tid] = tmp;
  
  __syncthreads();
  
  if (tid == 0)
  {
    REAL sum = 0.0;
    for (int i=0; i<TPB; i++)
      sum += temp[i];
    
    atomicAdd(s, sum);
  }
}



template <typename REAL>
static inline REAL svm_cost(const svm_param_t<REAL> *restrict args)
{
  const int m = args->m;
  const int n = args->n;
  const REAL *const restrict x = args->x;
  const int *const restrict y = args->y;
  const REAL *const restrict w = args->w;
  REAL *const restrict s = args->s;
  REAL *const restrict work = args->work;
  int check;
  REAL J;
  REAL norm;
  REAL s_cpu;
  
  int nb = m / TPB;
  if (m % TPB)
    nb++;
  
  // J_local = 1/m * sum(hinge_loss(1.0 - y * (x %*% w)))
  norm = euc_norm_sq(args->handle, n, w);
  
  mvm(args->handle, m, n, x, w, work);
  
  cudaMemset(s, 0, sizeof(*s));
  hinge_loss_sum<<<nb, TPB>>>(s, m, y, work);
  
  cudaMemcpy(&s_cpu, s, sizeof(*s), cudaMemcpyDeviceToHost);
  J = ((REAL) 1.0/m) * s_cpu;
  
  // J = allreduce(J_local) + 1/m * 0.5 * norm2(w)
  check = allreduce1(&J, args->comm);
  MPI_CHECK(args->comm, check);
  
  J += ((REAL) 1.0/m) * 0.5 * norm;
  
  return J;
}



template <typename REAL>
static inline void svm_nmwrap(int n, point_t<REAL> *point, const void *arg)
{
  const svm_param_t<REAL> *restrict args = (const svm_param_t<REAL>*) arg;
  cudaMemcpy(args->w, point->x, n*sizeof(REAL), cudaMemcpyHostToDevice);
  point->fx = svm_cost(args);
  cudaMemcpy(point->x, args->w, n*sizeof(REAL), cudaMemcpyDeviceToHost);
}



template <typename REAL>
static inline void svm(const int m, const int n, const REAL *const restrict x,
  const int *const restrict y, REAL *const restrict w, MPI_Comm restrict comm,
  optimset_t<REAL> *const restrict optimset)
{
  svm_param_t<REAL> args;
  point_t<REAL> start, solution;
  
  
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate_v2(&handle);
  if (st != CUBLAS_STATUS_SUCCESS)
    error("cublasCreate() failed\n");
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  
  REAL *x_gpu;
  int *y_gpu;
  REAL *w_gpu;
  REAL *work_gpu;
  REAL *s_gpu;
  
  cudaMalloc(&x_gpu, m*n*sizeof(*x_gpu));
  cudaMalloc(&y_gpu, m*sizeof(*y_gpu));
  cudaMalloc(&w_gpu, n*sizeof(*w_gpu));
  cudaMalloc(&work_gpu, m*sizeof(*work_gpu));
  cudaMalloc(&s_gpu, sizeof(*s_gpu));
  
  if (x_gpu == NULL || y_gpu == NULL || w_gpu == NULL || work_gpu == NULL || s_gpu == NULL)
  {
    CUFREE(x_gpu);
    CUFREE(y_gpu);
    CUFREE(w_gpu);
    CUFREE(work_gpu);
    CUFREE(s_gpu);
    error("Unable to allocate device memory");
  }
  
  cudaMemcpy(x_gpu, x, m*n*sizeof(*x), cudaMemcpyHostToDevice);
  cudaMemcpy(y_gpu, y, m*sizeof(*y), cudaMemcpyHostToDevice);
  
  start.x = w;
  memset(w, 0, n*sizeof(*w));
  
  args.handle = handle;
  args.m = m;
  args.n = n;
  args.x = x_gpu;
  args.y = y_gpu;
  args.w = w_gpu;
  args.s = s_gpu;
  args.work = work_gpu;
  args.comm = comm;
  
  nelder_mead(n, &start, &solution, &svm_nmwrap, &args, optimset);
  
  for (int i=0; i<n; i++)
    w[i] = solution.x[i];
  
  cublasDestroy_v2(handle);
  
  cudaFree(x_gpu);
  cudaFree(y_gpu);
  cudaFree(w_gpu);
  cudaFree(work_gpu);
  
  free(solution.x);
}



extern "C" SEXP R_svm(SEXP x, SEXP y, SEXP maxiter, SEXP comm_)
{
  SEXP ret, ret_names, w, niters;
  MPI_Comm comm = get_mpi_comm_from_Robj(comm_);
  const int m = nrows(x);
  const int n = ncols(x);
  
  PROTECT(ret = allocVector(VECSXP, 2));
  PROTECT(ret_names = allocVector(STRSXP, 2));
  PROTECT(niters = allocVector(INTSXP, 1));
  
  SET_STRING_ELT(ret_names, 0, mkChar("w"));
  SET_STRING_ELT(ret_names, 1, mkChar("niters"));
  
  if (TYPEOF(x) == REALSXP)
  {
    PROTECT(w = allocVector(REALSXP, n));
    
    optimset_t<double> opts;
    set_nm_opts<double>(INTEGER(maxiter)[0], &opts);
    svm<double>(m, n, REAL(x), INTEGER(y), REAL(w), comm, &opts);
  }
  else if (TYPEOF(x) == INTSXP)
  {
    PROTECT(w = allocVector(INTSXP, n));
  
    optimset_t<float> opts;
    set_nm_opts<float>(INTEGER(maxiter)[0], &opts);
    svm<float>(m, n, FLOAT(x), INTEGER(y), FLOAT(w), comm, &opts);
  }
  
  SET_VECTOR_ELT(ret, 0, w);
  SET_VECTOR_ELT(ret, 1, niters);
  setAttrib(ret, R_NamesSymbol, ret_names);
  
  UNPROTECT(4);
  return ret;
}
