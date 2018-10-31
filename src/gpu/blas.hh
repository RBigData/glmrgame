#ifndef GLMRGAME_GPUBLAS_H_
#define GLMRGAME_GPUBLAS_H_


#include <cublas_v2.h>

// c    = A    *    b
//  mx1    mxn       nx1
static inline void mvm(cublasHandle_t handle, const int m, const int n, const double *const __restrict__ A, const double *const __restrict__ b, double *const __restrict__ c)
{
  const cublasOperation_t trans = CUBLAS_OP_N;
  const double alpha = 1.0;
  const double beta = 0.0;
  const int k = 1;
 
  cublasDgemm(handle, trans, trans, m, k, n, &alpha, A, m, b, n, &beta, c, m);
}


#endif

