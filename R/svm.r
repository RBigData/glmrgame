#' @useDynLib glmrgame R_svm
NULL



#' svm
#' 
#' Support vector machine.
#' 
#' @details
#' The optimization uses Nelder-Mead.
#' 
#' Both of \code{x} and \code{y} must be distributed in an identical fashion.
#' This means that the number of rows owned by each MPI rank should match, and
#' the data rows \code{x} and response rows \code{y} should be aligned.
#' Additionally, each MPI rank should own at least one row.  Ideally they should
#' be load balanced, so that each MPI rank owns roughly the same amount of data.
#' 
#' @section Communication:
#' The communication consists of an allreduce of 1 double (the local
#' cost/objective function value) at each iteration of the optimization.
#' 
#' @param x,y
#' The input data \code{x} and response \code{y}.  Each must be a shaq, and
#' each must be distributed in an identical fashion.  See the details section
#' for more information.
#' @param maxiter
#' The maximum number of iterations.
#' 
#' @return
#' TODO
#' 
#' @examples
#' \dontrun{
#' library(kazaam)
#' comm.set.seed(1234, diff=TRUE)
#' 
#' x = ranshaq(rnorm, 10, 3)
#' y = ranshaq(function(i) sample(0:1, size=i, replace=TRUE), 10)
#' 
#' fit = svm(x, y)
#' comm.print(fit)
#' 
#' finalize()
#' }
#' 
#' @references
#' Efron, B. and Hastie, T., 2016. Computer Age Statistical Inference (Vol. 5).
#' Cambridge University Press.
#' 
#' @export
svm_game = function(x, y, maxiter=500)
{
  kazaam:::check.is.shaq(x)
  kazaam:::check.is.shaq(y)
  kazaam:::check.is.posint(maxiter)
  
  maxiter = as.integer(maxiter)
  comm_ptr = pbdMPI::get.mpi.comm.ptr(.pbd_env$SPMD.CT$comm)
  
  if (!is.double(DATA(x)))
    storage.mode(x@Data) = "double"
  
  if (!is.integer(DATA(y)))
    storage.mode(y@Data) = "integer"
  
  ret = .Call(R_svm, DATA(x), DATA(y), maxiter, comm_ptr)
  ret
}
