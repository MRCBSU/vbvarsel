import numpy as np
import math
from scipy.special import digamma

def calcAlphak(NK, alpha0, T):
    #A57
    ''' Function to find the updated variational parameter alphaK, i.e., the concentration parameter for 
            Dirichelet posterior distribution on the mixture proportions

    Params:
        NK: number of observations assigned to each cluster K
        alpha0: Prior coefficient count
        T: annealing temperature (T = 1 when annealing is not used)
    
    '''
    alpha = (NK + alpha0 + T - 1) / T
    return alpha

def calcAkj(K, J, C, NK, a0, T):
    '''Function to calculate the updated variational parameter akj

    Params:
        K: int -> Maximum number of clusters
        J: int -> number of covariates
        C: array[] -> covariate selection indicators
        NK: number of observations assigned to each cluster K
        a0: degrees of freedom, for the Gamma prior
        T: annealing temperature (T = 1 when annealing is not used)

    Returns:
        akj: updated variational parameter for the degrees of freedom of the posterior Gamma distribution
    
    '''
    #A60
    C = np.array(C).reshape(1, J)
    NK = np.array(NK).reshape(K, 1)
    akj = (C * NK / 2 + a0 + T - 1)/T
    return akj

def calcXd(Z, X):
    '''Function to find Xd

    Params:
        Z: np.ndarray
            cluster assignment matrix
        X: np.ndarray
            input data
    
    Returns:
        xd
    
    '''
    N= X.shape[0]
    N1= Z.shape[0]
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
    '''Function to calculate S

    Params:
        Z:
        X:
        xd:
    
    Returns:
        S
    
    '''
    K = Z.shape[1]
    XDim = X.shape[1]
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

def calcbetakj(K, XDim, C, NK, beta0, T):
    #A58
    '''Function to calculate the updated variational parameter betaKJ.

    Params:
        K: maximum number of clusters
        XDim: number of variables (columns)
        C: covariate selection indicators
        NK: number of observations assigned to each cluster K
        beta0: shrinkage parameter of the Gaussian conditional prior
        T: annealing temperature
    
    Returns:
        beta: updated variational shrinkage parameter for the Gaussian conditional posterior
    '''
    C = np.array(C).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    beta = (C * NK + beta0)/T
    return beta

def calcM(K, XDim, beta0, m0, NK, xd, betakj, C, T):
    #A59
    '''Function to calculate the updated variational parameter M.

    Params:
        K: maximum number of clusters
        XDim: number of variables (columns)
        beta0: shrinkage parameter of the Gaussian conditional prior
        m0: prior cluster means
        NK: number of observations assigned to each cluster K
        xd:
        betakj: updated variational shrinkage parameter for the Gaussian conditional posterior
        C: covariate selection indicators
        T: annealing temperature

    Returns:   
        m: updated variational cluster means
    
    '''
    m0 = np.array(m0).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    C = np.array(C).reshape(1, XDim)
    
    m = (beta0 * m0 + C * NK * xd) / (betakj * T)
    return m

def calcB(W0, xd, K, m0, XDim, beta0, S, C, NK, T):
    '''Function to calculate the updated variational parameter B

    Params:
        W0:
        xd:
        k:
        m0:
        XDim:
        beta0:
        S:
        C:
        NK:
        T:

    Returns:
        M:
    
    '''
    epsilon = 1e-8  # small constant to avoid division by zero
    M = np.zeros((K, XDim, XDim))
    Q0 = xd - m0[None, :]
    for k in range(K):
        M[k, np.diag_indices(XDim)] = 1/(W0 + epsilon) + NK[k]*np.diag(S[k])*C
        M[k, np.diag_indices(XDim)] += ((beta0*NK[k]*C) / (beta0 + C*NK[k])) * Q0[k]**2
    M = M/(2*T)
    return M

def calcDelta(C, d, T):
    '''Function to calculate the updated variational parameter Delta

    Params:
        C: covariate selection indicators
        d: shape of the Beta prior on the covariate selection probabilities
        T: annealing temperature
    
    Returns:
        Array of calculate annealing
    '''
    return np.array([(c + d + T - 1)/(2*d + 2*T - 1) for c in C])

def expSigma(X, XDim, betak, m, b, a, C):
    '''Function to calculate expSigma

    Params:
        X:
        XDim:
        betak:
        m:
        b:
        a:
        C: covariate selection indicators

    Returns:
        s:
    
    '''
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

    return s

def expPi(alpha0,NK):
    #A45
    '''Function to calculate expPi
    
    Params:
        alpha0: concentration of the Dirichlet prior on the mixture weights π
        NK: number of expected observations associated with the Kth component

    Returns:
        pik:
    '''
    alphak = alpha0 + NK
    pik = digamma(alphak) - digamma(alphak.sum())
    return pik

def expTau(b, a, C):
    '''Function to calculate expTau

    Params:
        b:
        a:
        C:

    Returns:
        invc:
    
    '''
    b = np.array(b)
    a = np.array(a)
    C = np.array(C)

    dW = np.diagonal(b, axis1=1, axis2=2)
    ld = np.where(dW > 1e-30, np.log(dW), 0.0)

    s = (digamma(a) - ld) * C

    invc = np.sum(s, axis=1)

    return invc.tolist()

def calcF0(X, XDim, sigma_0, mu_0, C):
    '''Function to calculate F0

    Params:
        X:
        XDim:
        sigma_0:
        mu_0:
        C:
    
    Returns:
        f0:
    
    '''
    C = np.array(C).reshape(1, XDim)
    sigma_0 = np.array(sigma_0).reshape(1, XDim)
    mu_0 = np.array(mu_0).reshape(1, XDim)
    
    f = np.array([[normal(xj, mu_0j, sigma_0j) for xj, mu_0j, sigma_0j in zip(x, mu_0[0], sigma_0[0])] for x in X])
    f0 = np.sum(f * (1 - C), axis=1)

    return f0

def calcZ(exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, N, K, C, T):
    '''Function to the updated variational parameter Z

    Params:
        exp_ln_pi: expected natural log of pi
        exp_ln_gam: expected natural log of gamma
        exp_ln_mu: expected natural log of mu
        f0: 
        N: the nth observation
        K: the kth cluster of the observation
        C: covariate selection indicator
        T: annealing temperature

    Returns:
        Z1:
    
    '''
    Z = np.zeros((N,K)) #ln Z
    for k in range(K):
        Z[:,k] = (exp_ln_pi[k] + 0.5*exp_ln_gam[k] - 0.5*sum(C)*np.log(2*math.pi) - 0.5*exp_ln_mu[:,k] + f0)/T
    #normalise ln Z:
    Z -= np.reshape(Z.max(axis=1),(N,1))
    Z = np.exp(Z) / np.reshape(np.exp(Z).sum(axis=1), (N,1))
    
    return Z

def normal(x, mu, sigma):
    '''Function to get a normal distribution

    Params:
        x: data
        mu: mean of the normal distribution
        sigma: standard deviation of the normal distribution
    
    Returns:
        n:
    
    '''
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    n = p * np.exp(-0.5 * ((x - mu)**2)/(sigma**2))
    return n

def calcexpF(X, b, a, m, beta, Z):
    '''Function to calculate expF

    Params:
        X:
        b:
        a:
        m:
        beta:
        Z:
    
    Returns:
        expF: intermediate factor to calculate the updated covariate selection indicators

    '''
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
    '''Function to calculate exp of F0

    Params:
        X:
        N:
        K:
        XDim:
        Z:
        sigma_0:
        mu_0:

    Returns:
        expF0: intermediate factor to calculate the updated covariate selection indicators
    
    '''
    expF0 = np.zeros(XDim)
    for j in range(XDim):
        s = 0
        for n in range(N):
            f = normal(X[n,j], mu_0[j], sigma_0[j])
            if f>1e-30: ld = np.log(f)
            else: ld = 0.0
            for k in range(K):
                s += Z[n,k]*ld
                
        expF0[j] = s
    return expF0

def calcN1(C, d, expF, T):
    #A53
    '''Function to calculate N1
    
    Params:
        C:
        d:
        expF:
        T:
    
    Returns:
        N1, lnN1: intermediate factors to calculate the updated covariate selection indicators
    '''
    expDelta = digamma((C + d + T - 1)/T) - digamma((2*d + 2*T - 1)/T)
    lnN1 = (expDelta + expF)/(T)
    N1 = np.exp(lnN1)
    return N1 , lnN1
    
def calcN2(C, d, expF0, T):
    #A54
    '''Function to calculate N2

    Params:
        C:
        d:
        expF0:
        T:

    Returns:
        N2, lnN2: intermediate factors to calculate the updated covariate selection indicators
    '''
    expDelta = digamma((T - C + d)/T) - digamma((2*d + 2*T - 1)/T)
    lnN2 = (expDelta + expF0)/(T)
    N2 = np.exp(lnN2)
    return N2 , lnN2
    
def calcC(XDim, 
                   N, 
                   K, 
                   X, 
                   b, 
                   a, 
                   m, 
                   beta, 
                   d,
                   C, 
                   Z, 
                   sigma_0,
                   mu_0,
                   T, 
                   trick=False
                   ):
    '''Function to calculate the updated variational parameter C, covariate selection indicators

    Params:
        XDim:
        N:
        K:
        X:
        b:
        a:
        m:
        beta:
        d:
        C:
        Z:
        sigma_0:
        mu_0:
        T:
        trick: bool -> whether to use or not a mathematical trick to avoid numerical errors

    Returns:
        C0:
    '''
    expF = calcexpF(X, b, a, m, beta, Z)
    expF0 = calcexpF0(X, N, K, XDim, Z, sigma_0, mu_0)
    N1, lnN1 = calcN1(C, d, expF, T)
    N2, lnN2 = calcN2(C, d, expF0, T)
    epsilon = 1e-40 
    
    if not trick:
        C0 = np.where(N1 > 0, N1 / (N1 + N2), 0)
    else:
        B = np.maximum(lnN1, lnN2)
        t1 = np.exp(lnN1 - B)
        t2 = np.exp(lnN2 - B)

        C0 = np.where(t1 > 0, t1 / (t1 + t2 + epsilon), 0)
        
    return C0


if __name__ == "__main__":
    
    x = expSigma([[1,1,1,1],[0,0,0,0]], 1, 1, 1, [[[1,1,0],[1,0,1]]], [[[1,1,0],[1,0,1]]], 1)

    print(x)
    print(type(x))
    
    pass