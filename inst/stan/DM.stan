data{
	int<lower=1> N; // total number of observations
	int<lower=2> D; // number of categories
	int<lower=0> n[N]; // number of trials
	int<lower=2> K; // number of predictor levels
	matrix[N,K] X; // predictor design matrix
	//matrix[N,D] Y; // response variable
	int Y[N,D];
	real sd_prior; // Prior standard deviation
}

transformed data{
	matrix[N,D] Y_real;
	
	for(i in 1:N){
		Y_real[i,] = to_row_vector(Y[i,]);
	}
}

parameters {
	matrix[D-1,K] beta_raw; // coefficients (raw)
	real<lower=0> aplus;
}

transformed parameters{
	matrix[D,K] beta; // coefficients
	matrix[N,D] mu;
	matrix[N,D] logits;
	
	for (l in 1:K) {
		beta[D,l] = 0.0;
	}

	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta[k,l] = beta_raw[k,l];
		}
	}
	
	for (i in 1:N) {
		for (m in 1:D){
			logits[i,m] = X[i,] * transpose(beta[m,]);
		}
		mu[i,] = to_row_vector(softmax(to_vector(logits[i,])));
	}
	
	
}

model {
// prior:

    aplus ~ gamma(1*0.001, 0.001);
	
	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta_raw[k,l] ~ normal(0,sd_prior);
		}
	}
	
// likelihood
	for (i in 1:N) {
		real temp;
		temp = 0.0;
		for(t in 1:D){
			temp = temp + lgamma((aplus*mu[i,t])+Y[i,t]) - lgamma(aplus*mu[i,t])  - lgamma(Y[i,t]+1);
		}
		target += lgamma(n[i]+1) + lgamma(aplus) - lgamma(aplus+n[i]) + temp;
	}
}


generated quantities{
	vector[N] log_lik;

// likelihood
	for (i in 1:N) {
		real temp;
		temp = 0.0;
		for(t in 1:D){
			temp = temp + lgamma((aplus*mu[i,t])+Y[i,t]) - lgamma(aplus*mu[i,t])  - lgamma(Y[i,t]+1);
		}
		log_lik[i] = lgamma(n[i]+1) + lgamma(aplus) - lgamma(aplus + n[i]) + temp;
	}
}
