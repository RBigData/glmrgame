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

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads). You will also need the development version of the float package:

```r
remotes::install_github("wrathematics/float")
```
