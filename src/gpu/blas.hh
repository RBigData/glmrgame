#ifndef GLMRGAME_GPUBLAS_H_
#define GLMRGAME_GPUBLAS_H_


#include <cublas_v2.h>


// ----------------------------------------------------------------------------
// ||x||^2
// ----------------------------------------------------------------------------
static inline float euc_norm_sq(cublasHandle_t handle, const int n, const float *const __restrict__ x)
{
  float norm;
  cublasStatus_t ret = cublasSnrm2(handle, n, x, 1, &norm);
  
  return norm;
}

static inline double euc_norm_sq(cublasHandle_t handle, const int n, const double *const __restrict__ x)
{
  double norm;
  cublasStatus_t ret = cublasDnrm2(handle, n, x, 1, &norm);
  
  return norm;
}



// ----------------------------------------------------------------------------
// c    = A    *    b
//  mx1    mxn       nx1
// ----------------------------------------------------------------------------
static inline void mvm(cublasHandle_t handle, const int m, const int n, const float *const __restrict__ A, const float *const __restrict__ b, float *const __restrict__ c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const float alpha = 1.0;
  const float beta = 0.0;
  const int k = 1;
 
  cublasSgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
}

static inline void mvm(cublasHandle_t handle, const int m, const int n, const double *const __restrict__ A, const double *const __restrict__ b, double *const __restrict__ c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const double alpha = 1.0;
  const double beta = 0.0;
  const int k = 1;
 
  cublasDgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
}


#endif
