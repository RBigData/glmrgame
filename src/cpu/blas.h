#ifndef GLMRGAME_CPUBLAS_H_
#define GLMRGAME_CPUBLAS_H_


void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
  const int *k, const double *restrict alpha, const double *restrict a,
  const int *lda, const double *restrict b, const int *ldb,
  const double *beta, double *restrict c, const int *ldc);

// c    = A    *    b
//  mx1    mxn       nx1
static inline void mvm(const int m, const int n, const double *const restrict A, const double *const restrict b, double *const restrict c)
{
  const char trans = 'N';
  const double alpha = 1.0;
  const double beta = 0.0;
  const int k = 1;
  
  dgemm_(&trans, &trans, &m, &k, &n, &alpha, A, &m, b, &n, &beta, c, &m);
}


#endif
