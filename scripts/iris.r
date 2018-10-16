suppressMessages(library(glmrgame))

fit = function(x, y)
{
  t1 = comm.timer(w1 <- svm_game(x, y))
  comm.print(t1)
  t2 = comm.timer(w2 <- svm(x, y))
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



data(iris)
is.setosa = expand((iris[, 5] == "setosa")*2 - 1)
iris = expand(as.matrix(iris[, -5]))
iris = cbind(shaq(1, nrow=nrow(iris), ncol=1), iris)

w = fit(iris, is.setosa)
p1 = get_pred(iris, is.setosa, w$w1)
p2 = get_pred(iris, is.setosa, w$w2)
comm.print(p1)
comm.print(p2)

finalize()
