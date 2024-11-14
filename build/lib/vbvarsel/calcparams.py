import numpy as np
import math
from scipy.special import digamma


def calcAlphak(NK: float, alpha0: float, T: float) -> float:
    # A57
    """Function to find the updated variational parameter alphaK, i.e., the concentration parameter for
            Dirichelet posterior distribution on the mixture proportions

    Params
    ------
        NK: float
            number of observations assigned to each cluster K
        alpha0: float
            Prior coefficient count
        T: float
            annealing temperature

    Returns
    -------
        alpha: np.ndarray[float]
            calculated alpha value
    """
    alpha = (NK + alpha0 + T - 1) / T
    return alpha


def calcAkj(
    K: int, J: int, C: np.ndarray[float], NK: float, a0: float, T: float
) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter akj

    Params
    ------
        K: int
            Maximum number of clusters
        J: int
            number of covariates
        C: np.ndarray[float]
            covariate selection indicators
        NK: float
            number of observations assigned to each cluster K
        a0: float
            degrees of freedom, for the Gamma prior
        T: float
            annealing temperature

    Returns
    -------
        akj: float
            updated variational parameter for the degrees of freedom of the
            posterior Gamma distribution

    """
    # A60
    C = np.array(C).reshape(1, J)
    NK = np.array(NK).reshape(K, 1)
    akj = (C * NK / 2 + a0 + T - 1) / T
    return akj


def calcXd(Z: np.ndarray, X: np.ndarray[float]) -> np.ndarray[float]:
    """Function to find Xd

    Params
    ------
        Z: np.ndarray
            cluster assignment matrix
        X: np.ndarray[float]
            2-D array of normalised data

    Returns
    -------
        xd: np.ndarray[float]
            Array of values

    """
    N = X.shape[0]
    N1 = Z.shape[0]
    NK = Z.sum(axis=0)
    assert N == N1

    # Add a small constant to avoid division by zero
    epsilon = 1e-10

    # Vectorized multiplication and sum
    xd = (Z.T @ X) / (NK[:, None] + epsilon)

    # Safe divide: replace inf and nan with 0
    xd = np.nan_to_num(xd)

    return xd


def calcS(
    Z: np.ndarray, X: np.ndarray[float], xd: np.ndarray[float]
) -> np.ndarray[float]:
    """Function to calculate S

    Params
    ------
        Z: np.ndarray
            cluster assignment matrix
        X: ndarray[float]
        xd: ndarray[float]

    Returns
    -------
        S: ndarray[float]

    """
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


def calcbetakj(
    K: int, XDim: int, C: np.ndarray[int], NK: float, beta0: float, T: float
) -> np.ndarray[float]:
    # A58
    """Function to calculate the updated variational parameter betaKJ.

    Params
    ------
        K: int
            maximum number of clusters
        XDim: int
            number of variables (columns)
        C: np.ndarray[int]
            covariate selection indicators
        NK: float
            number of observations assigned to each cluster K
        beta0: float
            shrinkage parameter of the Gaussian conditional prior
        T: float
            annealing temperature

    Returns
    -------
        beta: np.ndarray[float]
            updated variational shrinkage parameter for the Gaussian conditional
            posterior
    """
    C = np.array(C).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    beta = (C * NK + beta0) / T
    return beta


def calcM(
    K: int,
    XDim: int,
    beta0: float,
    m0: float,
    NK: float,
    xd: np.ndarray[float],
    betakj: np.ndarray[float],
    C: np.ndarray[int],
    T: float,
) -> np.ndarray[float]:
    # A59
    """Function to calculate the updated variational parameter M.

    Params
    ------
        K: int
            maximum number of clusters
        XDim: int
            number of variables (columns)
        beta0: float
            shrinkage parameter of the Gaussian conditional prior
        m0: float
            prior cluster means
        NK: float
            number of observations assigned to each cluster K
        xd: np.ndarray[float]
            calculated xd
        betakj: np.ndarray[float]
            updated variational shrinkage parameter for the Gaussian conditional posterior
        C: np.ndarray[int]
            covariate selection indicators
        T: float
            annealing temperature

    Returns
    -------
        m: np.ndarray[float]
            updated variational cluster means

    """
    m0 = np.array(m0).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    C = np.array(C).reshape(1, XDim)

    m = (beta0 * m0 + C * NK * xd) / (betakj * T)
    return m


def calcB(W0, xd, K, m0, XDim, beta0, S, C, NK, T) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter B

    Params
    ------
        W0: np.ndarray[float]
            2-D array with diaganal 1s rest 0s
        xd: np.ndarray[float]
            array of values generated from `calcXd`
        K: int
            hyperparameter k1, the number of clusters
        m0: np.ndarray[int]
            array of 0s with same shape as test data
        XDim: int
            number of columns
        beta0: float
            Shrinkage parameter of the Gaussian conditional prior on the cluster
            mean.
        S: list[np.ndarray[float]]
            list of ndarrays with of float values detrived from `calcS`
        C: np.ndarray[float]
            ndarray of 1s
        NK: float
            number of observations assigned to each cluster K

        T: float
            annealing temperature

    Returns
    -------
        M: np.ndarray[float]
            calculated variational parameter B
    """
    epsilon = 1e-8  # small constant to avoid division by zero
    M = np.zeros((K, XDim, XDim))
    Q0 = xd - m0[None, :]
    for k in range(K):
        M[k, np.diag_indices(XDim)] = 1 / (W0 + epsilon) + NK[k] * np.diag(S[k]) * C
        M[k, np.diag_indices(XDim)] += ((beta0 * NK[k] * C) / (beta0 + C * NK[k])) * Q0[
            k
        ] ** 2
    M = M / (2 * T)
    return M


def calcDelta(C: np.ndarray[float], d: int, T: float) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter Delta

    Params
    ------
        C: np.ndarray[float]
            covariate selection indicators
        d: int
            shape of the Beta prior on the covariate selection probabilities
        T: float
            annealing temperature

    Returns
    -------
        np.ndarray of calculate annealing
    """
    return np.array([(c + d + T - 1) / (2 * d + 2 * T - 1) for c in C])


def expSigma(
    X: np.ndarray[float],
    XDim: int,
    betak: float,
    m: np.ndarray[float],
    b: np.ndarray[float],
    a: np.ndarray[float],
    C: np.ndarray[float],
) -> float:
    """Function to calculate expSigma

    Params
    ------
        X: np.ndarray[float]
            2-D normalised array of data
        XDim: int
            number of columns
        betak: float
            calcultaed betakj value
        m: np.ndarray[float]
            calculated m value
        b: np.ndarray[float]
            calculated bkj value
        a: np.ndarray[float]
            calculated akj value
        C: np.ndarray[int]
            covariate selection indicators

    Returns
    -------
        s: float

    """
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


def expPi(alpha0: float, NK: float) -> np.ndarray[float]:
    # A45
    """Function to calculate expPi

    Params
    ------
        alpha0: float
            concentration of the Dirichlet prior on the mixture weights π
        NK: float
            number of expected observations associated with the Kth component

    Returns
    -------
        pik: np.ndarray[float]
            expected value of pi

    """
    alphak = alpha0 + NK
    pik = digamma(alphak) - digamma(alphak.sum())
    return pik


def expTau(
    b: np.ndarray[float], a: np.ndarray[float], C: np.ndarray[int]
) -> list[float]:
    """Function to calculate expTau

    Params
    ------
        b: np.ndarray[float]
            calcbkj value
        a: np.ndarray[float]
            calcalphakj value
        C: np.ndarray[int]
            covariate selection indicators

    Returns
    -------
        invc: list[float]
            The calculated expected Tau value


    """
    b = np.array(b)
    a = np.array(a)
    C = np.array(C)

    dW = np.diagonal(b, axis1=1, axis2=2)
    ld = np.where(dW > 1e-30, np.log(dW), 0.0)

    s = (digamma(a) - ld) * C

    invc = np.sum(s, axis=1)

    return invc.tolist()


def calcF0(
    X: np.ndarray[float],
    XDim: int,
    sigma_0: np.ndarray[float],
    mu_0: np.ndarray[float],
    C: np.ndarray[float],
) -> float:
    """Function to calculate F0

    Params
    ------
        X: np.ndarray[float]
            2-D array of normalised data
        XDim: int
            number of columns
        sigma_0: np.ndarray[float]
            Paramater estimate for Phi0j as MLE
        mu_0: np.ndarray[float]
            Paramater estimate for Phi0j as MLE
        C: np.ndarray[int]
            covariate selection indicators

    Returns
    -------
        f0: float
            calculated f0 parameter

    """
    C = np.array(C).reshape(1, XDim)
    sigma_0 = np.array(sigma_0).reshape(1, XDim)
    mu_0 = np.array(mu_0).reshape(1, XDim)

    f = np.array(
        [
            [
                normal(xj, mu_0, sigma_0)
                for xj, mu_0, sigma_0 in zip(x, mu_0[0], sigma_0[0])
            ]
            for x in X
        ]
    )
    f0 = np.sum(f * (1 - C), axis=1)

    return f0


def calcZ(
    exp_ln_pi: np.ndarray[float],
    exp_ln_gam: np.ndarray[float],
    exp_ln_mu: np.ndarray[float],
    f0: float,
    N: int,
    K: int,
    C: np.ndarray[float],
    T: float,
) -> np.ndarray[float]:
    """Function to the updated variational parameter Z

    Params
    ------
        exp_ln_pi: np.ndarray[float]
            expected natural log of pi
        exp_ln_gam: np.ndarray[float]
            expected natural log of gamma
        exp_ln_mu: np.ndarray[float]
            expected natural log of mu
        f0: float
            calculated f0 value
        N: int
            the nth observation
        K: int
            the kth cluster of the observation
        C: np.ndarray[int]
            covariate selection indicator
        T: float
            annealing temperature

    Returns
    -------
        Z1: np.ndarray[float]

    """
    Z = np.zeros((N, K))  # ln Z
    for k in range(K):
        Z[:, k] = (
            exp_ln_pi[k]
            + 0.5 * exp_ln_gam[k]
            - 0.5 * sum(C) * np.log(2 * math.pi)
            - 0.5 * exp_ln_mu[:, k]
            + f0
        ) / T
    # normalise ln Z:
    Z -= np.reshape(Z.max(axis=1), (N, 1))
    Z = np.exp(Z) / np.reshape(np.exp(Z).sum(axis=1), (N, 1))

    return Z


def normal(
    x: np.ndarray[float], mu: float, sigma: np.ndarray[float]
) -> np.ndarray[float]:
    """Function to get a normal distribution

    Params
    ------
        x: np.ndarray[float]
            2-D array of normalised data
        mu: float
            mean of the normal distribution
        sigma: np.ndarray[float]
            standard deviation of the normal distribution

    Returns
    -------
        n: np.ndarray[float]
            Array with normalised distribution

    """
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    n = p * np.exp(-0.5 * ((x - mu) ** 2) / (sigma**2))
    return n


def calcexpF(
    X: np.ndarray[float],
    b: np.ndarray[float],
    a: np.ndarray[float],
    m: np.ndarray[float],
    beta: np.ndarray[float],
    Z: np.ndarray[float],
) -> float:
    """Function to calculate expF

    Params
    ------
        X: np.ndarray[float]
            2-D array of normalised data
        b: np.ndarray[float]
            calculated bkj value
        a: np.ndarray[float]
            calculated akj value
        m: np.ndarray[float]
            calculated m value
        beta: np.ndarray[float]
            calculated betakj value
        Z: np.ndarray
            cluster assignment matrix

    Returns
    -------
        expF: float
            intermediate factor to calculate the updated covariate selection indicators

    """
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

    B0 = (X_exp - m_exp) ** 2
    B1 = (B0 * a_exp) / (b_exp)
    t3 = B1 + 1 / (beta_exp)

    s = Z_exp * (-np.log(2 * np.pi) + t2 - t3)
    expF = np.sum(s, axis=(0, 1)) * 0.5

    return expF


def calcexpF0(
    X: np.ndarray[float],
    N: int,
    K: int,
    XDim: int,
    Z: np.ndarray,
    sigma_0: np.ndarray[float],
    mu_0: np.ndarray[float],
) -> np.ndarray[float]:
    """Function to calculate exp of F0

    Params
    ------
        X: np.ndarray
            2-D array of normalised data
        N: int
            the nth observation
        K: int
            the kth cluster of the observation
        XDim: int
            number of columns
        Z: np.ndarray
            cluster assignment matrix
        sigma_0: np.ndarray[float]
            n-dim array of squared sigma values
        mu_0: np.ndarray[float]
            n-dim array of squared mu values

    Returns
    -------
        expF0: np.ndarray[float]
            intermediate factor to calculate the updated covariate selection indicators

    """
    expF0 = np.zeros(XDim)
    for j in range(XDim):
        s = 0
        for n in range(N):
            f = normal(X[n, j], mu_0[j], sigma_0[j])
            if f > 1e-30:
                ld = np.log(f)
            else:
                ld = 0.0
            for k in range(K):
                s += Z[n, k] * ld

        expF0[j] = s
    return expF0


def calcN1(C: np.ndarray[int], d: int, expF: float, T: float) -> tuple:
    # A53
    """Function to calculate N1

    Params
    ------
        C: np.ndarray[int]
            covariate selection indicator
        d: int
            Shape parameter of the Beta distribution on the probability.
        expF: float
            intermediate factor to calculate the updated covariate selection indicators
        T: float
            Annealing temperature

    Returns
    -------
        N1, lnN1:
            intermediate factors to calculate the updated covariate selection indicators
    """
    expDelta = digamma((C + d + T - 1) / T) - digamma((2 * d + 2 * T - 1) / T)
    lnN1 = (expDelta + expF) / (T)
    N1 = np.exp(lnN1)
    return N1, lnN1


def calcN2(C: np.ndarray[int], d: int, expF0: float, T: float) -> tuple:
    # A54
    """Function to calculate N2

    Params
    ------
        C: np.ndarray[int]
            covariate selection indicator
        d: int
            Shape parameter of the Beta distribution on the probability.
        expF0: float
            intermediate factor to calculate the updated covariate selection indicators
        T: float
            Annealing temperature

    Returns
    -------
        N2, lnN2: intermediate factors to calculate the updated covariate selection indicators
    """
    expDelta = digamma((T - C + d) / T) - digamma((2 * d + 2 * T - 1) / T)
    lnN2 = (expDelta + expF0) / (T)
    N2 = np.exp(lnN2)
    return N2, lnN2


def calcC(
    XDim: int,
    N: int,
    K: int,
    X: np.ndarray[float],
    b: np.ndarray[float],
    a: np.ndarray[float],
    m: np.ndarray[float],
    beta: np.ndarray[float],
    d: int,
    C: np.ndarray[float],
    Z: np.ndarray,
    sigma_0: np.ndarray[float],
    mu_0: np.ndarray[float],
    T: float,
    trick: bool = False,
) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter C, covariate selection indicators

    Params
    ------
        XDim: int
            number of columns
        N: int
            the nth observation
        K: int
            the kth cluster of the observation
        X: np.ndarray[float]
            2-D array of normalised data
        b: np.ndarray[float]
            calculated bkj value
        a: np.ndarray[float]
            calculated akj value
        m: np.ndarray[float]
            calculated m value
        beta: np.ndarray[float]
            calculated beta value
        d: int
            Shape parameter of the Beta distribution on the probability.
        C: np.ndarray[float]
            covariate selection indicator
        Z: np.ndarray
            cluster assignment matrix
        sigma_0: np.ndarray[float]
            n-dim array of squared sigma values
        mu_0: np.ndarray[float]
            n-dim array of squared mu values
        T: float
            Annealing temperature
        trick: bool
            whether or not to use a mathematical trick to avoid numerical errors

    Returns
    -------
        C0: np.ndarray[float]
            calculated variational parameter C
    """
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


# if __name__ == "__main__":

#     x = expSigma(
#         [[1, 1, 1, 1], [0, 0, 0, 0]],
#         1,
#         1,
#         1,
#         [[[1, 1, 0], [1, 0, 1]]],
#         [[[1, 1, 0], [1, 0, 1]]],
#         1,
#     )

#     # print(x)
#     # print(type(x))

#     pass
