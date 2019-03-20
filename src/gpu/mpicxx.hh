#ifndef GLMRGAME_MPICXX_H_
#define GLMRGAME_MPICXX_H_


#define OMPI_SKIP_MPICXX 1
#include <mpi.h>


static inline int allreduce1(float *const restrict J, const MPI_Comm comm)
{
  return MPI_Allreduce(MPI_IN_PLACE, J, 1, MPI_FLOAT, MPI_SUM, comm);
}

static inline int allreduce1(double *const restrict J, const MPI_Comm comm)
{
  return MPI_Allreduce(MPI_IN_PLACE, J, 1, MPI_DOUBLE, MPI_SUM, comm);
}


#endif
