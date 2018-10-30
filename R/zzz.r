#' @useDynLib glmrgame R_glmrgame_init
.onLoad = function(libname, pkgname)
{
  comm_ptr = pbdMPI::get.mpi.comm.ptr(.pbd_env$SPMD.CT$comm)
  .Call(R_glmrgame_init, comm_ptr)
  
  invisible()
}
