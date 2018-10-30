suppressMessages(library(glmrgame))
comm.set.seed(1234, diff=TRUE)
method = commandArgs(trailingOnly=TRUE)
method = tolower(method[[1]])
if (!isTRUE(method == "gpu") && !isTRUE(method == "cpu"))
  comm.stop("launch via 'Rscript x.r gpu' or Rscript x.r cpu'")


#m = 200000
#m = 5726623 # 32 gb summit/3
m = 8589934 # 32 gb summit/2
n = 250

size = comm.size()
m.local = m %/% size
rem = m - m.local*size
if (comm.rank()+1L < rem)
  m.local = m.local + 1



generator = function(m, n, means=c(0, 2), method="gpu")
{
  # if (method == "gpu")
  #   generator = curand::rnorm
  # else
  generator = stats::rnorm
  
  response = c(-1L, 1L)
  
  x = matrix(0.0, m, n+1)
  y = integer(m)
  
  time = comm.timer({
    for (i in 1:m)
    {
      group = sample(2, size=1)
      x[i, ] = c(1, generator(n, mean=means[group]))
      y[i] = response[group]
    }
  })
  
  list(x=x, y=y, time=time)
}

fit = function(x, y, maxiter=500, method="gpu")
{
  if (method == "gpu")
  {
    time = comm.timer(w <- glmrgame::svm_game(x, y, maxiter))
    w = w[[1]]
  }
  else
  {
    time = comm.timer(w <- kazaam::svm(x, y, maxiter))
    w = w$par
  }
  
  list(w=w, time=time)
}

get_pred = function(iris, is.setosa, w)
{
  pred = sign(DATA(iris) %*% w)
  acc.local = sum(pred == DATA(is.setosa))
  acc = allreduce(acc.local) / nrow(iris) * 100
  acc
}





.pbd_env$SPMD.CT$print.quiet = TRUE

comm.cat(paste("###", method, "---", size, "resources \n"))

data = generator(m.local, n, method=method)
comm.cat(paste("data time:", data$time["max"], "\n"))
x = shaq(data$x, nrows=m, ncols=n+1)
y = shaq(data$y, nrows=m, ncols=1)

w = fit(x, y, method=method)
p = get_pred(x, y, w$w)

comm.cat(paste("svm time: ", w$time["max"], "\n"))
comm.cat(paste("accuracy: ", p, "\n"))
comm.cat("\n")


finalize()
