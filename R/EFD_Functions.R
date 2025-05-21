# Functions to work with the EFD(M) distribution:




rEFD <- function(n, alpha, p, tau) {
  D <- length(alpha)
  multin <- matrix(rmultinom(n, 1, p), ncol=D, byrow=TRUE)
  x <- matrix(NA, nrow=n, ncol=D, byrow=TRUE)
  for (i in 1:n) { for (j in 1:D) {
    if (multin[i,j] == 1) x[i,j] <- rgamma(1, alpha[j] + tau[j]) else x[i,j] <- rgamma(1, alpha[j])
  }
  }
  somma <- apply(x, 1, sum)
  return(x/as.vector(somma))
}

to.alpha <- function(mu, w, p, aplus){
  aplus*((mu-t(diag(p)%*%w))/as.numeric(1- p%*%w))
}

to.tau <- function(w, aplus) t(aplus*(w/(1-w)))

w_norm2w <- function(w_norm, mu, p){
  apply(mu,1, function(x) w_norm*pmin(1, x/p))
}

rEFD.alternative <- function(n, mu, p, w_norm, aplus) {
  D <- length(p)
  w <- w_norm2w(w_norm, mu, p)
  tau <- to.tau(w,aplus)
  alpha <- to.alpha(mu,w,p,aplus)
  multin <- matrix(rmultinom(n, 1, p), ncol=D, byrow=TRUE)
  y <- t(apply(alpha+tau*multin, 1, function(x) rdirichlet(1,x)))
  return(y)
}
