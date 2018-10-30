#include <cuda_runtime.h>
#include <mpi.h>

extern "C" {
#include <Rinternals.h>
#include "../mpi_utils.h"
}


extern "C" SEXP R_glmrgame_init(SEXP comm_)
{
  int ngpus;
  int rank;
  int id;
  
  MPI_Comm *comm = get_mpi_comm_from_Robj(comm_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  cudaGetDeviceCount(&ngpus);
  
  id = rank % ngpus;
  cudaSetDevice(id);
  
#ifdef GLMRGAME_DEBUG
  printf("ngpus=%d rank=%d id=%d\n", ngpus, rank, id);
#endif
  
  return R_NilValue;
}
