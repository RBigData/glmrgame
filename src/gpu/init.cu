#include <cuda_runtime.h>
#include <Rinternals.h>
#include "../mpi_utils.h"


extern "C" SEXP R_glmrgame_init(SEXP comm_)
{
  int ngpus;
  int rank;
  
  MPI_Comm comm = get_mpi_comm_from_Robj(comm_);
  MPI_Comm_rank(comm, &rank);
  
  cudaGetDeviceCount(&ngpus);
  
  int id = rank % ngpus;
  cudaSetDevice(id);
  
  // printf("ngpus=%d rank=%d id=%d\n", ngpus, rank, id);
  return R_NilValue;
}
