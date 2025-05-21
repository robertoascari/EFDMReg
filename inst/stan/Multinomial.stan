data{
	int<lower=1> N; // total number of observations
	int<lower=2> D; // number of categories
	int<lower=2> K; // number of predictor levels
	matrix[N,K] X; // predictor design matrix
	//matrix[N,D] Y; // response variable
	int Y[N,D];
	real sd_prior; // Prior standard deviation
}

parameters {
	matrix[D-1,K] beta_raw; // coefficients (raw)
	//real theta;
}

transformed parameters{
	//real exptheta = exp(theta);
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
	
	for (n in 1:N) {
		for (m in 1:D){
			logits[n,m] = X[n,] * transpose(beta[m,]);
		}
		mu[n,] = to_row_vector(softmax(to_vector(logits[n,])));
	}
	
	
}

model {
// prior:
    //theta ~ normal(0,sd_prior);
	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta_raw[k,l] ~ normal(0,sd_prior);
		}
	}
// likelihood
	for (n in 1:N) {
		//transpose(Y[n,]) ~ dirichlet(transpose(mu[n,]) * exptheta);
		Y[n,] ~ multinomial(transpose(mu[n,]));
	}
}

generated quantities{
	vector[N] log_lik;
	for(n in 1:N){
		//log_lik[n] = dirichlet_lpdf(transpose(Y[n,]) | transpose(mu[n,]) * exptheta);
		log_lik[n] = multinomial_lpmf(Y[n,] | transpose(mu[n,]));
	}
}
