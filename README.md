# EFDMReg Package

<!-- badges: start -->

<!-- badges: end -->

First of all, import some necessary libraries.
```r
#library(FlexReg)
library(LearnBayes)
library(ggplot2)
library(loo)
library(rstan)
rstan_options(auto_write = TRUE)
Sys.setenv(LOCAL_CPPFLAGS = '-march=native')
```

Then, import stan models and create objects that can be used by the `rstan::sampling()` function.

```r
wd_stanmodels <- "inst/stan/"

Mult <- rstan::stan_model(file=paste(wd_stanmodels,"/Multinomial.stan",sep=""))
DM <- rstan::stan_model(file=paste(wd_stanmodels,"/DM.stan",sep=""))
FDM <- rstan::stan_model(file=paste(wd_stanmodels,"/FDM2.stan",sep=""))
EFDM <- rstan::stan_model(file=paste(wd_stanmodels,"/EFDM_hyper_w.stan",sep=""))
```

# Generating the data:

```r
source("R/EFD_Functions.R")

set.seed(396)
N <- 300 # Sample size
D <- 4
n <- 1000

# Design matrix:
X <- cbind(rep(1,N), runif(N,-.5,.5))

# Parameters' true value
aplus <- 50
w_norm <- c(.6,.2,.9,.4)
p <- c(.25,.3,.2,.25)

#              beta0  beta1
beta <- matrix(c( -.5,  1.8,
                  1.5, -2.5,
                  3,    -2,
                  0,    0),
               ncol = D)

a.parz <- exp(X%*%beta)
mu <- apply(a.parz,2,function(x) x/rowSums(a.parz))


pi_EFD <- rEFD.alternative(N,mu,p,w_norm,aplus)
Y <- matrix(NA, ncol=D, nrow=N)

for(i in 1:N){
  Y[i,] <- t(rmultinom(1, n, pi_EFD[i,]))
}

# Plotting the generated data:
par(mfrow=c(2,2))
for(j in 1:D)
  plot(Y[,j]/n~ X[,2], pch=20)
par(mfrow=c(1,1))
```

# Fitting stan models.

First of all, we need to create a list containing all the data required by the stan models' `data` block  

```r
# Stan setting:
n.iter <- 8000
nchain <- 1
warmup = 0.5*n.iter

# Data required by stan models:
data.stan <- list(
  N = N,
  n = rep(n, N),
  D = D,
  K = dim(X[,])[2],
  X = X,
  Y = Y,
  sd_prior = 50,
  w_hyper = rep(1, D)
)

# Fitting the stan models:
fit.Mult <- rstan::sampling(
  object = Mult,
  data = data.stan,
  warmup = warmup, iter = n.iter,
  cores = 1, thin=1, chains = nchain,
  pars=c("beta_raw", "log_lik"),
  refresh = n.iter/100
)

fit.DM <- rstan::sampling(
  object = DM,
  data = data.stan,
  warmup = warmup, iter = n.iter,
  cores = 1, thin=1, chains = nchain,
  pars=c("beta_raw", "aplus", "log_lik"),
  refresh = n.iter/100
)

fit.FDM <- rstan::sampling(
  object = FDM,
  data = data.stan,
  warmup = warmup, iter = n.iter,
  cores = 1, thin=1, chains = nchain,
  pars=c("beta_raw", "aplus", "p", "w_norm", "log_lik"),
  refresh = n.iter/100
)

fit.EFDM <- rstan::sampling(
  object = EFDM,
  data = data.stan,
  warmup = warmup, iter = n.iter,
  cores = 1, thin=1, chains = nchain,
  pars=c("beta_raw", "aplus", "p", "w_norm", "log_lik"), refresh = n.iter/100
)
```
We can use the fitted models to explore posterior distributions:

```r
rstan::summary(fit.EFDM, pars=c("beta_raw", "aplus", "p", "w_norm"))$summary

beta_chain <- rstan::extract(fit.EFDM, pars="beta_raw")[[1]]
dim(beta_chain)

# Comparing the true and estimated values:
t(apply(beta_chain, 2:3, mean))
beta

hist(rstan::extract(fit.EFDM, pars = "aplus")[[1]], prob = T, main = "Posterior distribution of aplus")
abline(v = aplus, col = 2, lty = "dashed", lwd = 2)
```


