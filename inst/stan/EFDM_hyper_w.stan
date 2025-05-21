data{
	int<lower=1> N; // total number of observations
	int<lower=2> D; // number of categories
	int<lower=0> n[N]; // number of trials
	int<lower=1> K; // number of predictor levels
	matrix[N,K] X; // predictor design matrix
	//matrix[N,D] Y; // response variable
	int Y[N,D];
	real sd_prior; // Prior standard deviation
	//real<lower=0> kappa;
	//real<lower=0> g;
	vector<lower=0>[D] w_hyper;
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
	simplex[D] p;
	vector<lower=0,upper=1>[D] w_norm;
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
	p ~ dirichlet(rep_vector(1, D));
  //aplus ~ gamma(kappa*g, g);
  aplus ~ gamma(1*0.001, 0.001);
  
  for(r in 1:D){
    w_norm[r] ~ beta(w_hyper[r], w_hyper[r]);
  }
	
	
	for (k in 1:(D-1)) {
		for (l in 1:K) {
			beta_raw[k,l] ~ normal(0,sd_prior);
		}
	}
	
// likelihood
	for (i in 1:N) {
		real temp;
		real temp2;
		vector[D] alpha;
		vector[D] Y_temp;

		vector[D] tau;
		temp = 0.0;
		temp2 = 0.0;
		
		for(j in 1:D){
		  temp2=temp2+p[j]*w_norm[j]*fmin(1.0, mu[i,j]/p[j]);
		}
		
		for(j in 1:D){//j rappresenta r
			alpha[j] = aplus*((mu[i,j] - p[j]*w_norm[j]*fmin(1.0, mu[i,j]/p[j]))/(1-temp2));
			tau[j] = aplus*(w_norm[j]*fmin(1.0, mu[i,j]/p[j])/(1-w_norm[j]*fmin(1.0, mu[i,j]/p[j])));
			
			temp = temp + lgamma(alpha[j]+Y[i,j]) - lgamma(alpha[j])  - lgamma(Y[i,j]+1);//produttoria tra parentesi formula 2.44
			Y_temp[j] = log(p[j]) + lgamma(aplus+ tau[j]) - lgamma(aplus+n[i] + tau[j]) + lgamma(alpha[j]) + lgamma(alpha[j]+tau[j]+Y[i,j]) - lgamma(alpha[j]+tau[j]) - lgamma(alpha[j]+Y[i,j]);
		}
		  target += lgamma(n[i]+1) + temp + log_sum_exp(Y_temp);
	}
}


generated quantities {
	vector[N] log_lik;

// likelihood
	for (i in 1:N) {
		real temp;
		real temp2;
		vector[D] alpha;
		vector[D] Y_temp;

		vector[D] tau;
		temp = 0.0;
		temp2 = 0.0;
		
		for(j in 1:D){
		  temp2=temp2+p[j]*w_norm[j]*fmin(1.0, mu[i,j]/p[j]);
		}
		
		for(j in 1:D){//j rappresenta r
			alpha[j] = aplus*((mu[i,j] - p[j]*w_norm[j]*fmin(1.0, mu[i,j]/p[j]))/(1-temp2));
			tau[j] = aplus*(w_norm[j]*fmin(1.0, mu[i,j]/p[j])/(1-w_norm[j]*fmin(1.0, mu[i,j]/p[j])));
			
			temp = temp + lgamma(alpha[j]+Y[i,j]) - lgamma(alpha[j])  - lgamma(Y[i,j]+1);//produttoria tra parentesi formula 2.44
			Y_temp[j] = log(p[j]) + lgamma(aplus+ tau[j]) - lgamma(aplus+n[i] + tau[j]) + lgamma(alpha[j]) + lgamma(alpha[j]+tau[j]+Y[i,j]) - lgamma(alpha[j]+tau[j]) - lgamma(alpha[j]+Y[i,j]);
		}
		log_lik[i] = lgamma(n[i]+1) + temp + log_sum_exp(Y_temp);
	}
}

