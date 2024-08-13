from scipy.special import digamma, psi, gammaln, multigammaln, gamma
from scipy.stats import beta, bernoulli, multivariate_normal, multivariate_t

import math
import numpy as np

# ELBO COMPUTATION

def ln_pi(alpha,k):
        return digamma(alpha[k])-digamma(sum(alpha))
    
    
def ln_precision(a,b):
    if b>1e-30: ld = np.log(b)
    else: ld = 0.0
    return psi(a) - ld


def log_resp_annealed(exp_ln_gam, exp_ln_mu, f0, N, K, C, T):
    log_resp = np.zeros((N,K)) #ln Z
    for k in range(K):
        log_resp[:,k] = ( 0.5*exp_ln_gam[k] - 0.5*sum(C)*np.log(2*math.pi) - 0.5*exp_ln_mu[:,k] + f0)/T
    return log_resp


def ln_delta_annealed(C, j, d, T):
    return digamma((C[j] + d + T - 1.)/T) - digamma((2*d + 2*T - 1.)/T)


def ln_delta_minus_annealed(C, j, d, T):
    return digamma((T - C[j] + d)/T) - digamma((2*d + 2*T - 1.)/T)


def entropy_wishart(k, j, b, a): 
    if b[k][j,j]>1e-30: ld = np.log(b[k][j,j])
    else: ld = 0.0
    return gammaln(a[k][j]) - (a[k][j]-1)*digamma(a[k][j]) - ld + a[k][j]


# ANNEALING (irrelevant here)
# classical geometric schedule T(k) = T0 * alpha^k
# T0 is the initial temperature
def geometric_schedule(T0, alpha, k, n):
    if k < n:
        return T0 * (alpha ** k)
    else:
        return 1
    
# classical harmonic schedule T(k) = T0 / (1 + alpha * k)
# T0 is the initial temperature
def harmonic_schedule( T0, alpha, k, n):
    return T0 / (1 + alpha * k)

def elbo_computation(XDim, K, N, C, Z, d, delta, beta, beta0, alpha, alpha0, a, a0, b, b0, m, m0, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T=1):  
        
    # E[ln p(X|Z, μ, Λ)]
    def first_term(N, K, Z, alpha, C, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T):
        ln_resp = log_resp_annealed(exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, N, K, C, T)
        F2 = 0
        for n in range(N):
            for k in range(K):
                F2+= Z[n,k]*(ln_resp[n,k])
                
        return F2
        
    
    # E[ln p(Z|π)] 
    def second_term(N, K, Z, alpha):
        s=0
        for n in range(N):
            for k in range(K):
                s+=Z[n,k]*ln_pi(alpha,k)
        return s  
    
    # E[ln p(π)]
    def third_term(alpha0, K, alpha):
        a = gammaln(alpha0*K)-K*gammaln(alpha0)
        b = (alpha0-1)*sum([ln_pi(alpha,k) for k in range(K)])
        return a+b
    
    
    # E[ln p(μ, Λ)] 
    def fourth_term(K, XDim, beta0, beta, a0, a, b0, b, m, m0):
        t = 0
        for k in range(K):
            for j in range(XDim):
                F0 = 0.5*np.log(beta0/(2*math.pi)) + 0.5*ln_precision(a[k,j], b[k][j,j])
                if beta[k][j]>0: 
                    F1 = (beta0*a[k][j]/beta[k][j])*((m[k][j]-m0[j])**2) + beta0/beta[k][j] + b0[j,j]*a[k][j]/beta[k][j]
                else:
                    F1 = (beta0*a[k][j])*((m[k][j]-m0[j])**2) + beta0 + b0[j,j]*a[k][j]
                    
                F2 = - np.log(gamma(a0)) + a0*np.log(b0[j,j]) + (a0-2)*ln_precision(a[k,j], b[k][j,j])
                
                t += F0 - F1 + F2

        return t

    # E[ln p(𝛾,𝛿)] 
    def fifth_term(XDim, d, C, T):
        a = 0
        for j in range(XDim):
            F1 = (d+C[j]-1)*ln_delta_annealed(C, j, d, T)
            F2 = (d+ C[j])*ln_delta_minus_annealed(C, j, d, T)
            F3 = np.log(gamma(2*d)) - 2*np.log(gamma(d))
            a+= F1 + F2 + F3
        return a
    
    
    # E[ln q(Z)]
    def sixth_term(Z, N, K ):
        a = 0
        for n in range(N):
            for k in range(K):
                if Z[n, k]>1e-30: ld = np.log(Z[n, k])
                else: ld = 0.0
                a += Z[n, k]*ld
        return a
        
    
    
    # E[ln q(π)] 
    def seventh_term(alpha):
        a = sum([(alpha[k]-1)*ln_pi(alpha,k) for k in range(K)])
        b = gammaln(sum(alpha))-sum([gammaln(alpha[k]) for k in range(K)])
        return a+b
    
    
    # E[ln q(μ, Λ)] 
    def eighth_term(K, XDim, beta, a, b):
        t = 0
        for k in range(K):
            for j in range(XDim):
                t += .5*ln_precision(a[k,j], b[k][j,j])+0.5*np.log(beta[k][j]/(2*np.pi))-0.5-entropy_wishart(k, j, b, a)
        return t
    
        # E[ln q(𝛾,𝛿)] 
    def ninth_term(XDim, d, delta, C, T):
        F0 = 2*XDim*((d-1)*(digamma(d)-digamma(2*d))+np.log(gamma(2*d)) - 2*np.log(gamma(d)))
        F1 = 0
        for j in range(XDim):
            F1 += delta[j]*ln_delta_annealed(C, j, d, T) + (1-delta[j])*ln_delta_minus_annealed(C, j, d, T)
        
        return F0 + F1
    
    
    a_t=first_term(N, K, Z, alpha, C, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T)
    b_t=second_term(N, K, Z, alpha)
    c_t=third_term(alpha0, K, alpha)
    d_t=fourth_term(K, XDim, beta0, beta, a0, a, b0, b, m, m0)
    e_t=fifth_term(XDim, d, C, T)
    f_t=sixth_term(Z, N, K )
    g_t=seventh_term(alpha)
    h_t=eighth_term(K, XDim, beta, a, b)
    i_t=ninth_term(XDim, d, delta, C,  T)
    
    return a_t + b_t + c_t + d_t + e_t - (f_t + g_t + h_t + i_t)*T
