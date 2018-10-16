suppressMessages(library(glmrgame))
comm.set.seed(1234, diff=TRUE)

generator = function(m, n, means=c(0, 2))
{
  response = c(-1L, 1L)
  
  x = matrix(0.0, m, n+1)
  y = integer(m)
  
  for (i in 1:m)
  {
    group = sample(2, size=1)
    x[i, ] = c(1, rnorm(n, mean=means[group]))
    y[i] = response[group]
  }
  
  list(x=x, y=y)
}

fit = function(x, y, maxiter=500)
{
  t1 = comm.timer(w1 <- svm_game(x, y, maxiter))
  comm.print(t1)
  t2 = comm.timer(w2 <- svm(x, y, maxiter))
  comm.print(t2)
  
  w1 = w1[[1]]
  w2 = w2$par
  
  comm.print(head(w1))
  comm.print(head(w2))
  
  list(w1=w1, w2=w2)
}

get_pred = function(iris, is.setosa, w)
{
  pred = sign(DATA(iris) %*% w)
  acc.local = sum(pred == DATA(is.setosa))
  acc = allreduce(acc.local) / nrow(iris) * 100
  acc
}



m.local = 100000
m.local = 500
m = allreduce(m.local)
n = 250

data = generator(m.local, n)
x = shaq(data$x, nrows=m, ncols=n+1)
y = shaq(data$y, nrows=m, ncols=1)


w = fit(x, y)
p1 = get_pred(x, y, w$w1)
p2 = get_pred(x, y, w$w2)
comm.print(p1)
comm.print(p2)

finalize()
