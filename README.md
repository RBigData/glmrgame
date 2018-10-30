# glmrgame

* **Version:** 0.1-0
* **URL**: https://github.com/RBigData/glmrgame
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Author:** Drew Schmidt

🚨 Highly experimental 🚨

glmrgame (pronounced "glimmer-game") is a package for glm-like computations in R ("glimmer") run on gpu's (video game hardware), with computations distributed over MPI.


## Installation

The development version is maintained on GitHub:

```r
remotes::install_github("RBigData/glmrgame")
```

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads). You will also need the development version of the float, pbdMPI, and kazaam packages (and optionally the curand R package):

```r
remotes::install_github("wrathematics/float")
remotes::install_github("wrathematics/pbdMPI")
remotes::install_github("rbigdata/kazaam")
remotes::install_github("wrathematics/curand")
```



## Benchmarks

All timings are from:

* A DGX-1
* R 3.4.4
* OpenBLAS
* CUDA 9.0.176

Currently all data is generated on cpu. I plan to fix this soon for gpu benchmarks. I also, for reasons I don't wish to explain, am using 13 physical cores per gpu. The goal is to be faster even at this ratio. First we set:

```bash
export GPUS=2
export SCALER=13
```

We run the benchmarks via:

```bash
mpirun -np ${GPUS} Rscript x.r gpu
mpirun -np $(( ${GPUS}*${SCALER} )) Rscript x.r cpu
```

This gives the results:

```
### gpu --- 2 resources 
data time: 164.373 
svm time:  12.891 
accuracy:  100 

### cpu --- 26 resources (2 threads per rank)
data time: 15.77 
svm time:  111.725 
accuracy:  73.7294574940224 

### cpu --- 26 resources (1 thread per rank)
data time: 16.308 
svm time:  111.493 
accuracy:  73.7294574940224 
```

If we re-run with 1 gpu vs 13 cores (instead of 2 vs 26 above), we get:

```
### gpu --- 1 resources 
data time: 149.809 
svm time:  12.43 
accuracy:  73.7353046018747 

### cpu --- 13 resources (2 threads per rank)
data time: 14.857 
svm time:  97.658 
accuracy:  73.7324346688658 

### cpu --- 13 resources (1 thread per rank)
data time: 14.369 
svm time:  99.068 
accuracy:  73.7324346688658 
```
