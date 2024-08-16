from numpy import *
#import numpy as np
from matplotlib.pyplot import *
#import matplotlib.pyplot
import math
from numpy.random import choice, permutation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Ellipse
from numpy.linalg.linalg import inv, det
from scipy.special import digamma, psi, gammaln, multigammaln, gamma
from scipy.stats import beta, bernoulli, multivariate_normal, multivariate_t
import time
import itertools
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score
import seaborn as sns

# GETTING DATA

def normalise_data(data):
    
    # Compute the mean and standard deviation of each column
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Subtract the mean and divide by the standard deviation
    array_normalized = (data - mean) / std

    
    return array_normalized

def shuffle_data(data, out = 0):
   
    # Number of columns
    num_columns = shape(data)[1] - out
     
    # Generate a permutation of column indices
    shuffled_indices = np.random.permutation(num_columns)

    # Concatenate with the indices of the last 10 columns
    shuffled_indices = np.concatenate((shuffled_indices, np.arange(num_columns, shape(data)[1])))
    
    # Shuffle the columns of the matrix
    shuffled_data = data[:, shuffled_indices]
    
    return shuffled_data, shuffled_indices


def simulation_data_crook(n_observations, n_variables, n_relevant, mixture_proportions, means, variance_covariance_matrix):
    # Simulate relevant variables
    samples = []
    true_labels = []  # Store the true labels
    for _ in range(n_observations):
        # Select mixture component based on proportions
        component = np.random.choice([0, 1, 2], p=mixture_proportions)
        true_labels.append(component)  # Store the true label
        mean_vector = np.full(n_relevant, means[component])
        sample = np.random.multivariate_normal(mean_vector, variance_covariance_matrix)
        samples.append(sample)
    
    # Convert list of samples to numpy array
    relevant_variables = np.array(samples)

    # Simulate irrelevant variables
    n_irrelevant = n_variables - n_relevant
    irrelevant_variables = np.random.randn(n_observations, n_irrelevant)

    # Combine relevant and irrelevant variables
    data = np.hstack((relevant_variables, irrelevant_variables))
    
    # Shuffle the variables
    permutation = np.random.permutation(n_variables)
    shuffled_data = data[:, permutation]

    # Now data contains 100 observations with 200 variables, where the first n_relevant are drawn
    # from the Gaussian mixtures and the rest are irrelevant variables from standard Gaussian.

    return data, shuffled_data, true_labels, permutation

# PARAMETR UPDATES

def calcAlphak_annealed(NK, alpha0, T, K):
    alpha = (NK + alpha0 + T - 1) / T
    return alpha

def calcAkj_annealed(K, J, c, NK, a0, T):
    c = np.array(c).reshape(1, J)
    NK = np.array(NK).reshape(K, 1)
    a = (c * NK / 2 + a0 + T - 1)/T
    return a

def calcXd(Z, X):
    N, XDim = X.shape
    N1, K = Z.shape
    NK = Z.sum(axis=0)
    assert N == N1

    # Add a small constant to avoid division by zero
    epsilon = 1e-10

    # Vectorized multiplication and sum
    xd = (Z.T @ X) / (NK[:, None] + epsilon)

    # Safe divide: replace inf and nan with 0
    xd = np.nan_to_num(xd)

    return xd

def calcS(Z, X, xd):
    N, K = Z.shape
    N, XDim = X.shape
    NK = Z.sum(axis=0)

    # Initialize S as a list of zero matrices
    S = [np.zeros((XDim, XDim)) for _ in range(K)]
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-10

    # Calculate M for each k
    for k in range(K):
        diff = (X - xd[k]) ** 2
        S[k] = np.diag(Z[:, k] @ diff / (NK[k] + epsilon))

    return S

def calcbetakj_annealed(K, XDim, c, NK, beta0, T):
    c = np.array(c).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    beta = (c * NK + beta0)/T
    return beta

def calcM_annealed(K, XDim, beta0, m0, NK, xd, betakj, c, T):
    m0 = np.array(m0).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    c = np.array(c).reshape(1, XDim)
    
    m = (beta0 * m0 + c * NK * xd) / (betakj * T)
    return m

def calcB_annealed(W0,xd,K,m0,XDim,beta0,S, c, NK, T):
    epsilon = 1e-8  # small constant to avoid division by zero
    M = np.zeros((K, XDim, XDim))
    Q0 = xd - m0[None, :]
    for k in range(K):
        M[k, np.diag_indices(XDim)] = 1/(W0 + epsilon) + NK[k]*np.diag(S[k])*c
        M[k, np.diag_indices(XDim)] += ((beta0*NK[k]*c) / (beta0 + c*NK[k])) * Q0[k]**2
    M = M/(2*T)
    return M

def calcDelta_annealed(J, C, d, T):
    return np.array([(c + d + T - 1)/(2*d + 2*T - 1) for c in C])

def expSigma(X, XDim, NK, betak, m, b, xd, a, N, K, C):
    C = np.array(C).reshape(1, XDim)
    X_exp = np.expand_dims(X, axis=1)
    m_exp = np.expand_dims(m, axis=0)
    a_exp = np.expand_dims(a, axis=0)
    b_exp = np.diagonal(b, axis1=1, axis2=2)
    b_exp = np.expand_dims(b_exp, axis=0)
    betak_exp = np.expand_dims(betak, axis=0)

    B0 = X_exp - m_exp
    B1 = ((B0**2) * a_exp) / b_exp
    B1 += 1 / betak_exp
    s = np.sum(B1 * C, axis=2)

    Mu = s

    return Mu

def expPi(alpha0,NK):
    alphak = alpha0 + NK
    pik = digamma(alphak) - digamma(alphak.sum())
    return pik

def expTau(b, a, XDim, K, C):
    b = np.array(b)
    a = np.array(a)
    C = np.array(C)

    dW = np.diagonal(b, axis1=1, axis2=2)
    ld = np.where(dW > 1e-30, np.log(dW), 0.0)

    s = (digamma(a) - ld) * C

    invc = np.sum(s, axis=1)

    return invc.tolist()

def calcF0(X, N, XDim, sigma_0, mu_0, C):
    C = np.array(C).reshape(1, XDim)
    sigma_0 = np.array(sigma_0).reshape(1, XDim)
    mu_0 = np.array(mu_0).reshape(1, XDim)
    
    f = np.array([[normal(xj, mu_0j, sigma_0j) for xj, mu_0j, sigma_0j in zip(x, mu_0[0], sigma_0[0])] for x in X])
    f0 = np.sum(f * (1 - C), axis=1)

    return f0

def calcZ_annealed(XDim, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, N, K, C, T):  
    Z = zeros((N,K)) #ln Z
    for k in range(K):
        Z[:,k] = (exp_ln_pi[k] + 0.5*exp_ln_gam[k] - 0.5*sum(C)*log(2*pi) - 0.5*exp_ln_mu[:,k] + f0)/T
    #normalise ln Z:
    Z -= reshape(Z.max(axis=1),(N,1))
    Z1 = exp(Z) / reshape(exp(Z).sum(axis=1), (N,1))
    
    return Z1

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    n = p * np.exp(-0.5 * ((x - mu)**2)/(sigma**2))
    return n


def calcexpF(X, XDim, K, N, b, a, m, beta, Z):
    X_exp = X[:, None, :]
    m_exp = m[None, :, :]
    a_exp = a[None, :, :]
    b_diag = np.diagonal(b, axis1=1, axis2=2)  # extract the diagonal elements of b
    b_exp = b_diag[None, :, :]
    beta_exp = beta[None, :, :]
    Z_exp = Z[:, :, None]
    
    epsilon = 1e-30

    dW = np.where(b_exp > epsilon, np.log(b_exp), 0.0)
    t2 = digamma(a_exp) - dW

    B0 = (X_exp - m_exp)**2
    B1 = (B0 * a_exp) / (b_exp)
    t3 = B1 + 1/(beta_exp)

    s = Z_exp * (-np.log(2*np.pi) + t2 - t3)
    expF = np.sum(s, axis=(0, 1)) * 0.5

    return expF


def calcexpF0(X, N, K, XDim, Z, sigma_0, mu_0):
    expF0 = zeros(XDim)
    for j in range(XDim):
        s = 0
        for n in range(N):
            f = normal(X[n,j], mu_0[j], sigma_0[j])
            if f>1e-30: ld = log(f)
            else: ld = 0.0
            for k in range(K):
                s += Z[n,k]*ld
                
        expF0[j] = s
    return expF0

def calcN1_annealed(XDim, C, d, expF, T, N):
    expDelta = digamma((C + d + T - 1)/T) - digamma((2*d + 2*T - 1)/T)
    lnN1 = (expDelta + expF)/(T)
    N1 = exp(lnN1)
    return N1 , lnN1
    
def calcN2_annealed(XDim, C, d, expF0, T, N):
    expDelta = digamma((T - C + d)/T) - digamma((2*d + 2*T - 1)/T)
    lnN2 = (expDelta + expF0)/(T)
    N2 = exp(lnN2)
    return N2 , lnN2
    
def calcC_annealed(XDim, N, K, X, b, a, m, beta, d, C, Z, sigma_0, mu_0, T, trick=False):
    expF = calcexpF(X, XDim, K, N, b, a, m, beta, Z)
    expF0 = calcexpF0(X, N, K, XDim, Z, sigma_0, mu_0)
    N1, lnN1 = calcN1_annealed(XDim, C, d, expF, T, N)
    N2, lnN2 = calcN2_annealed(XDim, C, d, expF0, T, N)

    epsilon = 1e-40
    
    if not trick:
        C = np.where(N1 > 0, N1 / (N1 + N2), 0)
    else:
        B = np.maximum(lnN1, lnN2)
        t1 = exp(lnN1 - B)
        t2 = exp(lnN2 - B)

        C = np.where(t1 > 0, t1 / (t1 + t2 + epsilon), 0)
        
    return C


# ELBO COMPUTATION

def ln_pi(alpha,k):
        return digamma(alpha[k])-digamma(sum(alpha))
    
    
def ln_precision(a,b):
    if b>1e-30: ld = log(b)
    else: ld = 0.0
    return psi(a) - ld


def log_resp_annealed(exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, N, K, C, T):
    log_resp = zeros((N,K)) #ln Z
    for k in range(K):
        log_resp[:,k] = ( 0.5*exp_ln_gam[k] - 0.5*sum(C)*log(2*pi) - 0.5*exp_ln_mu[:,k] + f0)/T
    return log_resp


def ln_delta_annealed(C, j, d, T):
    return digamma((C[j] + d + T - 1.)/T) - digamma((2*d + 2*T - 1.)/T)


def ln_delta_minus_annealed(C, j, d, T):
    return digamma((T - C[j] + d)/T) - digamma((2*d + 2*T - 1.)/T)


def entropy_wishart(k, j, b, a): 
    if b[k][j,j]>1e-30: ld = log(b[k][j,j])
    else: ld = 0.0
    return gammaln(a[k][j]) - (a[k][j]-1)*digamma(a[k][j]) - ld + a[k][j]

def elbo_computation(XDim, K, N, C, Z, d, delta, beta, beta0, alpha, alpha0, a, a0, b, b0, m, m0, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T=1):  
    
    # E[ln p(X|Z, μ, Λ)]
    def first_term(N, K, Z, C, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T):
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
                if Z[n, k]>1e-30: ld = log(Z[n, k])
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

# MAIN RUN FUNCTION 
def run(X, K, alpha0, beta0, a0, m0, b0, d, C, threshold, T, annealing = "fixed", max_annealed_itr = 10, Ctrick = True):
    (N,XDim) = shape(X)
    
    
    # params:
    Z = np.array([random.dirichlet(ones(K)) for _ in range(N)])
    
    # parameter estimates for \Phi_{0j} precomputed as MLE
    mu_0 = zeros(XDim)
    sigma_sq_0 = ones(XDim)
    for j in range(XDim):
        mu_0[j] = sum(X[:, j])/N
        sigma_sq_0[j] = sum((X[:, j] - mu_0[j])**2)/N

    itr = 0
    
    lower_bound = []
    converged = False 

    # while itr < max_itr:
    while not converged:
        if annealing == "geometric":
            cooling_rate = (1 / T0) ** (1 / (max_annealed_itr - 1))
            T = geometric_schedule(T0, cooling_rate, itr, max_annealed_itr)
        elif annealing == "harmonic":
            cooling_rate = (T0 - 1) / max_annealed_itr
            T = harmonic_schedule(T0, cooling_rate, itr, max_annealed_itr)
        elif annealing == "fixed":
            T = T0
        
        NK = Z.sum(axis=0)
       
      
        #M-like-step
            
        alphak = calcAlphak_annealed(NK, alpha0, T, K)
        akj = calcAkj_annealed(K, XDim, C, NK, a0, T)
        xd = calcXd(Z,X)
        S = calcS(Z, X, xd)
        betakj = calcbetakj_annealed(K, XDim, C, NK, beta0, T)
        m = calcM_annealed(K, XDim, beta0, m0, NK, xd, betakj, C, T)
        bkj = calcB_annealed(b0, xd, K, m0, XDim, beta0, S, C, NK, T)
        delta = calcDelta_annealed(XDim, C, d, T)
        

        #E-like-step
        mu = expSigma(X, XDim, NK, betakj, m, bkj, xd, akj, N, K, C)
        invc = expTau(bkj, akj, XDim, K, C)
        pik = expPi(alpha0,NK) 
        f0 = calcF0(X, N, XDim, sigma_sq_0, mu_0, C)
        
        Z = calcZ_annealed(XDim, pik, invc, mu, f0, N, K, C, T) 
        C = calcC_annealed(XDim, N, K, X, bkj, akj, m, betakj, d, C, Z, sigma_sq_0, mu_0, T, trick = Ctrick)
        
        lb = elbo_computation(XDim, K, N, C, Z, d, delta, betakj, beta0, alphak, alpha0, akj, a0, bkj, b0, m, m0, pik, invc, mu, f0, T)
        lower_bound.append(lb)
        
        
        #Convergence criterion
        improve = (lb - lower_bound[itr-1]) if itr > 0 else lb
        if itr>0 and 0 < improve < threshold:
            print('Converged at iteration {}'.format(itr))
            converged = True 
            break
        
        
        itr += 1
            
    
    return m, S, invc, pik, Z, lower_bound, C, itr


def cluster_prediction(rsp):
    d = [np.argmax(r) for r in rsp]
    return d

#BEGIN
##VALUES FOR SIMULATION, DEFAULTS PROVIDED, GIVE USER OPTION TO OVERWRITE IF THEY SO CHOOSE
# SIMULATIONS 

# SIMULATIONS 

# Parameters for simulations
n_observations = [100, 1000] # number of observations to simulate
n_variables = 200 # number of variables to simulate
n_relevants = [10, 20, 50, 100]  # number of variables that are relevant
mixture_proportions = [0.5, 0.3, 0.2] # ~ proportion of observations in each cluster, length of the array defines number of simulated clusters
means = [0, 2, -2] # mean of the gaussian distribution for each cluster

# Prior settings and Hyperparameters for VBVarSel
    
threshold = 1e-1 # convergence threshold
K1 = 5 # maximum number of clusters

# prior coefficient count, or concentration parameter, for Dirichelet prior on the mixture proportions
# this parameter can be interpreted as pseudocounts, i.e. the effective prior number of observations associated with each mixture component.
alpha0 = 1/(K1) # or set to 0.01

# shrinkage parameter of the Gaussian conditional prior on the cluster mean
# influences the tightness and spread of the cluster, with smaller shrinkage leading to tighter clusters.
beta0 = (1e-3)*1. 

# degrees of freedom, for the Gamma prior on the cluster precision
# controls the shape of the Gamma distribution, the higher the degree of freedom, the more peaked
a0 = 3.
    
# shape parameter of the Beta distribution on the probability of selection
# d0 = 1, the Beta distribution turns into a uniform distribution.
d0 = 1

# maximum/starting annealing temperature
# T_max = 1 means no annealing
T0 = 1.


# maximum number of iterations
max_itr = 25

# number of models to run for model averaging
max_models = 10


convergence_ELBO = []
convergence_itr = []
clust_predictions = []
variable_selected = []
times = []
ARIs = []
n_relevant_var = []
n_obs = []

n_rel_rel = [] # correctly selected relevant variables
n_irr_irr = [] # correctly discarded irrelevant variables

#END

# goal is to do simultaneous clustering and variable selection

# user output
# array of variable selections `variable_selected`
# array of cluster_assignments `clust_predictions`
# uses variational inference vs mcmc 
# can run 1000s of observations and many variables very fast

for p in range(len(n_observations)):
    for n_rel in range(len(n_relevants)):
        for i in range(max_models):
            
            print("Model " + str(i))
            print("obs " + str(p))
            print("rel " + str(n_rel))
            print()
            
            n_relevant_var.append(n_relevants[n_rel])
            n_obs.append(n_observations[p])
    
            variance_covariance_matrix = np.identity(n_relevants[n_rel])
            X, X_shuffled, true_labels, index_array = simulation_data_crook(n_observations[p], n_variables, n_relevants[n_rel], mixture_proportions, means, variance_covariance_matrix)
    
            (N,XDim) = shape(X)
        
            C = np.ones(XDim)  
            W0 = (1e-1)*eye(XDim) #prior cov (bigger: smaller covariance)
            v0 = XDim + 2. 
            m0 = zeros(XDim) #prior mean
            for j in range(XDim):
                m0[j] = np.mean(X[:, j])
            
            delta0 = beta.rvs(d0, d0, size=XDim)
        
            # Measure the execution time of the following code
            start_time = time.time()
            mu, S, invc, pik, Z, lower_bound, Cs, itr = run(X_shuffled, K1, alpha0, beta0, v0, a0, m0, W0, d0, delta0, C, 
                                                           threshold, max_itr, T0)
            end_time = time.time()
            execution_time = end_time - start_time
    
            times.append(execution_time)
    
            NK = Z.sum(axis=0)
    
            convergence_ELBO.append(lower_bound[-1])
            convergence_itr.append(itr)
    
            clust_pred = [np.argmax(r) for r in Z]
            clust_predictions.append(clust_pred)
    
            ari = adjusted_rand_score(np.array(true_labels), np.array(clust_pred))
            ARIs.append(ari)
    
            original_order = np.argsort(index_array)
            var_selection_ordered = np.around(np.array(Cs)[original_order])
            variable_selected.append(var_selection_ordered)
            
            unique_counts, counts = np.unique(np.around(var_selection_ordered[:n_relevants[n_rel]]), return_counts=True)
       
            # Find the index of the specific element (e.g., element 0) in the unique_counts array
            element_to_extract = 1
            index_of_element = np.where(unique_counts == element_to_extract)[0]

            # Extract the counts of the specific element from the counts array
            counts_of_element = counts[index_of_element]
            n_rel_rel.append(counts_of_element[0])
    
        

            unique_counts, counts = np.unique(np.around(var_selection_ordered[n_relevants[n_rel]:]), return_counts=True)
        
            # Find the index of the specific element (e.g., element 0) in the unique_counts array
            element_to_extract = 0
            index_of_element = np.where(unique_counts == element_to_extract)[0]

            # Extract the counts of the specific element from the counts array
            counts_of_element = counts[index_of_element]
            n_irr_irr.append(counts_of_element[0])

#add as option to print or something along those lines w/option to save
#plotting? add way to change hyperparams
df_results = pd.DataFrame({'n_observations' : n_obs, 'n_relevant' : n_relevant_var, 'ARI' : ARIs, 'n_rel_rel' : n_rel_rel, 'n_irr_irr' : n_irr_irr, 
                        'time' : times, 'convergence ELBO' : convergence_ELBO, 'convergence itr' : convergence_itr})
    
# SAVING AS CSV
df_results.to_csv('simulation_studies_crook_results.csv', index=False)


if __name__ == "__main__":
    #BEGIN
##VALUES FOR SIMULATION, DEFAULTS PROVIDED, GIVE USER OPTION TO OVERWRITE IF THEY SO CHOOSE
# SIMULATIONS 

# SIMULATIONS 

# Parameters for simulations
    n_observations = [10] # number of observations to simulate
    n_variables = 200 # number of variables to simulate
    n_relevants = [10, 20, 50, 100]  # number of variables that are relevant
    mixture_proportions = [0.5, 0.3, 0.2] # ~ proportion of observations in each cluster, length of the array defines number of simulated clusters
    means = [0, 2, -2] # mean of the gaussian distribution for each cluster

    # Prior settings and Hyperparameters for VBVarSel
        
    threshold = 1e-1 # convergence threshold
    K1 = 5 # maximum number of clusters

    # prior coefficient count, or concentration parameter, for Dirichelet prior on the mixture proportions
    # this parameter can be interpreted as pseudocounts, i.e. the effective prior number of observations associated with each mixture component.
    alpha0 = 1/(K1) # or set to 0.01

    # shrinkage parameter of the Gaussian conditional prior on the cluster mean
    # influences the tightness and spread of the cluster, with smaller shrinkage leading to tighter clusters.
    beta0 = (1e-3)*1. 

    # degrees of freedom, for the Gamma prior on the cluster precision
    # controls the shape of the Gamma distribution, the higher the degree of freedom, the more peaked
    a0 = 3.
        
    # shape parameter of the Beta distribution on the probability of selection
    # d0 = 1, the Beta distribution turns into a uniform distribution.
    d0 = 1

    # maximum/starting annealing temperature
    # T_max = 1 means no annealing
    T_max = 1.

    # maximum number of iterations
    max_itr = 25

    # number of models to run for model averaging
    max_models = 10


    convergence_ELBO = []
    convergence_itr = []
    clust_predictions = []
    variable_selected = []
    times = []
    ARIs = []
    n_relevant_var = []
    n_obs = []

    n_rel_rel = [] # correctly selected relevant variables
    n_irr_irr = [] # correctly discarded irrelevant variables

    #END

    # goal is to do simultaneous clustering and variable selection

    # user output
    # array of variable selections `variable_selected`
    # array of cluster_assignments `clust_predictions`
    # uses variational inference vs mcmc 
    # can run 1000s of observations and many variables very fast

    for p in range(len(n_observations)):
        for n_rel in range(len(n_relevants)):
            for i in range(max_models):
                
                print("Model " + str(i))
                print("obs " + str(p))
                print("rel " + str(n_rel))
                print()
                
                n_relevant_var.append(n_relevants[n_rel])
                n_obs.append(n_observations[p])
        
                # variance_covariance_matrix = np.identity(n_relevants[n_rel])
                # X = simulation_data_crook(n_observations[p], n_variables, n_relevants[n_rel], mixture_proportions, means, variance_covariance_matrix)
                # print(X)