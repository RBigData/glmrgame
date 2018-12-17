#ifndef GLMRGAME_MPIUTILS_
#define GLMRGAME_MPIUTILS_


#include <mpi.h>
#include <Rinternals.h>


#define MPI_CHECK(comm, check) if (check != MPI_SUCCESS) R_mpi_throw_err(check, comm);

static inline void R_mpi_throw_err(int check, const MPI_Comm *comm)
{
  int rank;
  MPI_Comm_rank(*comm, &rank);
  if (rank == 0)
    error("MPI_Allreduce returned error code %d\n", check);
  else
    error(""); // FIXME
}



static inline MPI_Comm* get_mpi_comm_from_Robj(SEXP comm_)
{
  MPI_Comm *comm = (MPI_Comm*) R_ExternalPtrAddr(comm_);
  return comm;
}


#endif
