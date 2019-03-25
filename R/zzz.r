#' @useDynLib glmrgame R_glmrgame_init
.onLoad = function(libname, pkgname)
{
  s = search()
  if ("package:clustrgame" %in% s || "package:dimrgame" %in% s)
    return(invisible())
  
  comm_ptr = pbdMPI::get.mpi.comm.ptr(.pbd_env$SPMD.CT$comm)
  .Call(R_glmrgame_init, comm_ptr)
  
  invisible()
}
