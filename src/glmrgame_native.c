/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_svm(SEXP x, SEXP y, SEXP maxiter, SEXP comm_);

static const R_CallMethodDef CallEntries[] = {
  {"R_svm", (DL_FUNC) &R_svm, 4},
  {NULL, NULL, 0}
};

void R_init_kazaam(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
