#include <cublas_v2.h>
#include <mpi.h>

extern "C" {
  // #include <float/float32.h>
  // #include <float/slapack.h>
#include <Rinternals.h>
#include <stdlib.h>

#include "../common.h"
#include "../mpi_utils.h"
#include "../nm.h"
}

#include "blas.hh"
#include "cu_utils.hh"



typedef struct {
  cublasHandle_t handle;
  int m;
  int n;
  const double *__restrict__ x;
  const int *__restrict__ y;
  double *__restrict__ w;
  double *__restrict__ work;
  double *__restrict__ s;
  MPI_Comm *__restrict__ comm;
} svm_param_t;




static inline double euc_norm_sq(cublasHandle_t handle, const int n, const double *const __restrict__ x)
{
  double norm;
  cublasStatus_t ret = cublasDnrm2(handle, n, x, 1, &norm);
  
  return norm;
}



__global__ static void hinge_loss_sum(double *s, const int m, const int *const __restrict__ y, const double *const __restrict__ work)
{
  int tid = threadIdx.x;
  int i = tid + blockIdx.x*blockDim.x;
  
  if (i >= m)
    return;
  
  __shared__ double temp[TPB];
  
  double tmp = 1.0 - y[i]*work[i];
  if (tmp < 0.0)
    temp[tid] = 0.0;  
  else
    temp[tid] = tmp;
  
  __syncthreads();
  
  if (tid == 0)
  {
    double sum = 0.0;
    for (int i=0; i<TPB; i++)
      sum += temp[i];
    
    atomicAdd(s, sum);
  }
}



static inline double svm_cost(cublasHandle_t handle,
  const int m, const int n, const double *const __restrict__ x,
  const int *const __restrict__ y, const double *const __restrict__ w,
  double *const __restrict__ s, double *const __restrict__ work,
  const MPI_Comm *const __restrict__ comm)
{
  int check;
  double J;
  double norm;
  double s_cpu = 0.0;
  
  cudaMemset(s, 0, 1*sizeof(*s));
  
  // J_local = 1/m * sum(hinge_loss(1.0 - DATA(y)*matmult(DATA(x), w)))
  int nb = m / TPB;
  if (m % TPB)
    nb++;
  
  norm = euc_norm_sq(handle, n, w);
  
  mvm(handle, m, n, x, w, work);
  hinge_loss_sum<<<nb, TPB>>>(s, m, y, work);
  cudaMemcpy(&s_cpu, s, sizeof(*s), cudaMemcpyDeviceToHost);
  J = ((double) 1.0/m) * s_cpu;
  
  // J = allreduce(J_local) + 1/m * 0.5 * norm2(w)
  check = MPI_Allreduce(MPI_IN_PLACE, &J, 1, MPI_DOUBLE, MPI_SUM, *comm);
  MPI_CHECK(comm, check);
  
  J += ((double) 1.0/m) * 0.5 * norm;
  
  return J;
}



static inline void svm_nmwrap(int n, point_t *point, const void *arg)
{
  const svm_param_t *args = (const svm_param_t*) arg;
  cudaMemcpy(args->w, point->x, n*sizeof(double), cudaMemcpyHostToDevice);
  point->fx = svm_cost(args->handle, args->m, n, args->x, args->y, args->w, args->s, args->work, args->comm);
  cudaMemcpy(point->x, args->w, n*sizeof(double), cudaMemcpyDeviceToHost);
}



static inline void svm(const int m, const int n, const double *const __restrict__ x,
  const int *const __restrict__ y, double *const __restrict__ w, MPI_Comm *const __restrict__ comm,
  optimset_t *const __restrict__ optimset)
{
  svm_param_t args;
  point_t start, solution;
  
  
  cublasHandle_t handle;
  cublasStatus_t st = cublasCreate_v2(&handle);
  if (st != CUBLAS_STATUS_SUCCESS)
    error("cublasCreate() failed\n");
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
  
  double *x_gpu;
  int *y_gpu;
  double *w_gpu;
  double *work_gpu;
  double *s_gpu;
  
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
  svm(m, n, REAL(x), INTEGER(y), REAL(w), comm, &opts);
  
  UNPROTECT(4);
  return ret;
}
