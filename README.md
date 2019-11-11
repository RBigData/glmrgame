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

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads). You will also need the development version of the pbdMPI and kazaam packages (and optionally the curand R package):

```r
remotes::install_github("wrathematics/pbdMPI")
remotes::install_github("rbigdata/kazaam")
remotes::install_github("wrathematics/curand")
```

There is a reference cpu version of the package that you can build. However, this is not supported or recommended; please just use kazaam. But if you insist, you can install it via

```r
remotes::install_github("rbigdata/glmrgame", configure.args="--with-backend=CPU")
```


## Examples

```r
suppressMessages(library(kazaam))
suppressMessages(library(glmrgame))

data(iris)
is_setosa = expand((iris[, 5] == "setosa")*2 - 1)
iris = expand(as.matrix(iris[, -5]))
iris = cbind(shaq(1, nrow=nrow(iris), ncol=1), iris)

w_cpu = svm(iris, is_setosa)
w_gpu = svm_game(iris, is_setosa)

finalize()
```



## Benchmarks

All timings are from:

* A DGX-1
* R 3.4.4
* OpenBLAS
* CUDA 9.0.176

The benchmark requires of generating data from 2 different random normal distributions and using svm to classify the data. The data consists 251 columns (250 data + 1 intercept), and however many rows required for a desired total dataset size.

For reasons I don't wish to explain, I am using 13 cores for the cpu-only runs for every one gpu of the gpu runs. The goal is for the gpu runs to be faster even at this 13-to-1 ratio. First we set:

For a 16 GiB total problem size (distributed among the MPI ranks), we get:

```
### gpu --- 2 resources 
data time: 64.57 
svm time:  11.971 
accuracy:  100 

### cpu --- 26 resources (2 threads per rank)
data time: 16.197 
svm time:  111.445 
accuracy:  73.7294574940224 

### cpu --- 26 resources (1 thread per rank)
data time: 16.307 
svm time:  109.208 
accuracy:  73.7294574940224 
```

Data generation for the gpu case is done of the gpu using the [curand R package](https://github.com/wrathematics/curand). However, this approach requires many more memory operations, and the local problem size is 13x larger than each rank in the cpu-only case. Hence the relatively poor performance.

If we re-run with 1 gpu vs 13 cores (instead of 2 vs 26 above) on half the problem size (8 GiB total), we get:

```
### gpu --- 1 resources 
data time: 57.361 
svm time:  10.74 
accuracy:  100 

### cpu --- 13 resources (2 threads per rank)
data time: 15.057 
svm time:  96.4 
accuracy:  73.7324346688658 

### cpu --- 13 resources (1 thread per rank)
data time: 14.735 
svm time:  96.637 
accuracy:  73.7324346688658 
```

The scripts are in the `scripts/` directory of the source tree of glmrgame.
