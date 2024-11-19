import numpy as np
import math
from scipy.special import digamma


def calcAlphak(NK: float, alpha0: float, T: float) -> float:
    """Function to find the updated variational parameter alphaK, i.e., the concentration parameter for Dirichelet posterior distribution on the mixture proportions (A29/A57)

    Params
        NK: float
            Number of observations assigned to each cluster K
        alpha0: float
            Prior coefficient count, :func:`~vbvarsel.global_parameters.Hyperparameters.alpha0`
        T: float
            Annealing temperature
    Returns
        alphaK: np.ndarray[float]
            Calculated Alphak values

    """
    alphaK = (NK + alpha0 + T - 1) / T
    return alphaK


def calcAkj(
    K: int, J: int, C: np.ndarray[float], NK: float, a0: float, T: float
) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter akj (A35/A60)

    Params
        K: int
            The Kth cluster
        J: int
            Iteration count
        C: np.ndarray[float]
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        NK: float
            Number of observations assigned to each cluster K
        alpha0: float
            Degrees of freedom, for the Gamma prior, :func:`~vbvarsel.global_parameters.Hyperparameters.alpha0`
        T: float
            Annealing temperature
    Returns
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
    """Function to find Xd. (A37)

    Params
        Z: np.ndarray
            Latent cluster assignment matrix, :func:`~vbvarsel.calcparams.calcZ`
        X: np.ndarray[float]
            2-D array of normalised data
    Returns
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
    """Function to calculate Skj. (A38)

    Params
        Z: np.ndarray
            Latent cluster assignment matrix, :func:`~vbvarsel.calcparams.calcZ`
        X: ndarray[float]
            Shuffled array
        xd: ndarray[float]
            Variational paramater Xd, :func:`~vbvarsel.calcparams.calcXd`
    Returns
        S: ndarray[float]
            Calculated S variable parameter

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
    """Function to calculate the updated variational parameter betaKJ. (A33/A58)

    Params
        K: int
            The Kth cluster
        XDim: int
            number of variables (columns)
        C: np.ndarray[int]
            covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        NK: float
            Number of observations assigned to each cluster K
        beta0: float
            shrinkage parameter of the Gaussian conditional prior, :func:`~vbvarsel.global_parameters.Hyperparameters.beta0`
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`
    Returns
        beta: np.ndarray[float]
            Updated variational shrinkage parameter for the Gaussian conditional
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
    """Function to calculate the updated variational parameter Mkj (A34/A59).

    Params
        K: int
            The Kth cluster
        XDim: int
            number of variables (columns)
        beta0: float
            Shrinkage parameter of the Gaussian conditional prior, :func:`~vbvarsel.global_parameters.Hyperparameters.beta0`
        m0: float
            Prior cluster means
        NK: float
            Number of observations assigned to each cluster K
        xd: np.ndarray[float]
            Value of calculated variational parameter xd, , :func:`~vbvarsel.calcparams.calcXd`
        betakj: np.ndarray[float]
            Updated variational shrinkage parameter for the Gaussian conditional posterior
        C: np.ndarray[int]
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`
    Returns
        m: np.ndarray[float]
            Updated variational cluster means

    """
    m0 = np.array(m0).reshape(1, XDim)
    NK = np.array(NK).reshape(K, 1)
    C = np.array(C).reshape(1, XDim)

    m = (beta0 * m0 + C * NK * xd) / (betakj * T)
    return m


def calcB(W0, xd, K, m0, XDim, beta0, S, C, NK, T) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter B

    Params
        W0: np.ndarray[float]
            2-D array with diagonal 1s rest 0s
        xd: np.ndarray[float]
            Value of calculated variational parameter xd, :func:`~vbvarsel.calcparams.calcXd`
        K: int
            Hyperparameter k1, the number of clusters
        m0: np.ndarray[int]
            Array of 0s with same shape as test data
        XDim: int
            Number of variables (columns)
        beta0: float
            Shrinkage parameter of the Gaussian conditional prior on the cluster mean, :func:`~vbvarsel.global_parameters.Hyperparameters.beta0`
        S: list[np.ndarray[float]]
            Calculated value of variational paramater S, :func:`~vbvarsel.calcparams.calcS`
        C: np.ndarray[float]
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        NK: float
            Number of observations assigned to each cluster K
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`

    Returns
        B: np.ndarray[float]
            Calculated variational parameter B

    """
    epsilon = 1e-8  # small constant to avoid division by zero
    M = np.zeros((K, XDim, XDim))
    Q0 = xd - m0[None, :]
    for k in range(K):
        M[k, np.diag_indices(XDim)] = 1 / (W0 + epsilon) + NK[k] * np.diag(S[k]) * C
        M[k, np.diag_indices(XDim)] += ((beta0 * NK[k] * C) / (beta0 + C * NK[k])) * Q0[
            k
        ] ** 2
    B = M / (2 * T)
    return B


def calcDelta(C: np.ndarray[float], d: int, T: float) -> np.ndarray[float]:
    """Function to calculate the updated variational parameter Delta (A43/A62)

    Params
        C: np.ndarray[float]
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        d: int
            Shape of the Beta prior on the covariate selection probabilities, :func:`~vbvarsel.global_parameters.Hyperparameters.d0`
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`
    Returns: float
        Array of calculated variational parameter delta

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
    """Function to calculate the expected Sigma values. (A47/A48)

    Params
        X: np.ndarray[float]
            2-D normalised array of data
        XDim: int
            Number of variables (columns)
        betak: float
            Calculated value for the variational paramater betakj, :func:`~vbvarsel.calcparams.calcbetakj`
        m: np.ndarray[float]
            Calculated value for the variational paramater m, :func:`~vbvarsel.calcparams.calcM`
        b: np.ndarray[float]
            Calculated value for the variational paramater B, :func:`~vbvarsel.calcparams.calcB`
        a: np.ndarray[float]
            Calculated value for the variational paramater akj, :func:`~vbvarsel.calcparams.calcAkj`
        C: np.ndarray[int]
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
    Returns
        s: float
            Calculated expected sigma values

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
    """Function to calculate Expected Pi value (A45)

    Params
        alpha0: float
            Concentration of the Dirichlet prior on the mixture weights π, :func:`~vbvarsel.global_parameters.Hyperparamaters.alpha0`
        NK: float
            Number of expected observations associated with the Kth component
    Returns
        pik: np.ndarray[float]
            Expected values of pi

    """
    alphak = alpha0 + NK
    pik = digamma(alphak) - digamma(alphak.sum())
    return pik


def expTau(
    betakj: np.ndarray[float], akj: np.ndarray[float], C: np.ndarray[int]
) -> list[float]:
    """Function to calculate Expected Tau value (A47)

    Params
        betakj: np.ndarray
            Value for the calculated variational parameter betakj, :func:`~vbvarsel.calcparams.calcbetakj`
        akj: np.ndarray
            Value for the calculated variational parameter akj, :func:`~vbvarsel.calcparams.calcAkj`
        C: np.ndarray
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
    Returns
        invc: list[float]
            The calculated expected Tau values

    """
    b = np.array(betakj)
    a = np.array(akj)
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
    """Function to calculate F0 (A30?)

    Params
        X: np.ndarray
            2-D array of normalised data
        XDim: int
            Number of variables (columns)
        sigma_0: np.ndarray
            Paramater estimate for Phi0j as MLE
        mu_0: np.ndarray
            Paramater estimate for Phi0j as MLE
        C: np.ndarray
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
    Returns
        F0: np.ndarray
            Calculated value for variational parameter F0

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
    F0 = np.sum(f * (1 - C), axis=1)

    return F0


def calcZ(
    exp_ln_pi: np.ndarray[float],
    exp_ln_tau: np.ndarray[float],
    exp_ln_sigma: np.ndarray[float],
    f0: float,
    N: int,
    K: int,
    C: np.ndarray[float],
    T: float,
) -> np.ndarray[float]:
    """Function to the updated variational parameter Z, the latent cluster assignments (A23)

    Params
        exp_ln_pi: np.ndarray
            Expected natural log of pi, :func:`~vbvarsel.calcparams.expPi`
        exp_ln_tau: np.ndarray
            Expected natural log of tau, :func:`~vbvarsel.calcparams.expTau`
        exp_ln_sigma: np.ndarray
            Expected natural log of sigma, :func:`~vbvarsel.calcparams.expSigma`
        f0: float
            Calculated f0 value, :func:`~vbvarsel.calcparams.calcF0`
        N: int
            The nth observation
        K: int
            The kth cluster of the observation
        C: np.ndarray
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`
    Returns
        Z: np.ndarray
            Calculated variational parameter Z

    """
    Z = np.zeros((N, K))  # ln Z
    for k in range(K):
        Z[:, k] = (
            exp_ln_pi[k]
            + 0.5 * exp_ln_tau[k]
            - 0.5 * sum(C) * np.log(2 * math.pi)
            - 0.5 * exp_ln_sigma[:, k]
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
        x: np.ndarray
            2-D array of normalised data
        mu: float
            Mean of the normal distribution
        sigma: np.ndarray
            Standard deviation of the normal distribution
    Returns
        n: np.ndarray
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
    """Function to calculate expected F, an intermediate factor to calculate the updated covariate selection indicators

    Params
        X: np.ndarray
            2-D array of normalised data
        b: np.ndarray
            Value for the calculated variational parameter B, :func:`~vbvarsel.calcparams.calcB`
        a: np.ndarray
            Value for the calculated variational parameter akj, :func:`~vbvarsel.calcparams.calcAkj`
        m: np.ndarray
            Value for the calculated variational parameter m, :func:`~vbvarsel.calcparams.calcM`
        beta: np.ndarray
            Value for the calculated variational parameter betakj, :func:`~vbvarsel.calcparams.calcbetakj`
        Z: np.ndarray
            Latent cluster assignment matrix, :func:`~vbvarsel.calcparams.calcZ`
    Returns
        expF: float
            Intermediate factor to calculate the updated covariate selection indicators

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
    """Function to calculate expected F0, an intermediate factor to calculate the updated covariate selection indicators

    Params
        X: np.ndarray
            2-D array of normalised data
        N: int
            The nth observation
        K: int
            The kth cluster of the observation
        XDim: int
            Number of variables (columns)
        Z: np.ndarray
            Latent cluster assignment matrix, :func:`~vbvarsel.calcparams.calcZ`
        sigma_0: np.ndarray
            N-dim array of squared sigma values
        mu_0: np.ndarray
            N-dim array of squared mu values
    Returns
        expF0: np.ndarray
            Expected F0, an intermediate factor to calculate the updated covariate selection indicators

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
    """Function to calculate N1 (A41), a parameter for Cj (A40) in the Bernoulli distribution (A39)

    Params
        C: np.ndarray
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        d: int
            Shape parameter of the Beta distribution on the probability. :func:`~vbvarsel.global_parameters.Hyperparameters.d0`
        expF: float
            Intermediate factor to calculate the updated covariate selection indicators :func:`~vbvarsel.calcparams.calcexpF`
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`
    Returns
        N1, lnN1: tuple
            Intermediate factors to calculate the updated covariate selection indicators

    """
    expDelta = digamma((C + d + T - 1) / T) - digamma((2 * d + 2 * T - 1) / T)
    lnN1 = (expDelta + expF) / (T)
    N1 = np.exp(lnN1)
    return N1, lnN1


def calcN2(C: np.ndarray[int], d: int, expF0: float, T: float) -> tuple:
    # A54
    """Function to calculate N1 (A42), a parameter for Cj (A40) in the Bernoulli distribution (A39)

    Params
        C: np.ndarray
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        d: int
            Shape parameter of the Beta distribution on the probability. :func:`~vbvarsel.global_parameters.Hyperparameters.d0`
        expF0: float
            Intermediate factor to calculate the updated covariate selection indicators :func:`~vbvarsel.calcparams.calcexpF0`
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max` 
    Returns
        N2, lnN2: tuple
            Intermediate factors to calculate the updated covariate selection indicators

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
    """Function to calculate the updated variational parameter C (A40), the covariate selection indicators

    Params
        XDim: int
            Number of variables (columns)
        N: int
            the nth observation
        K: int
            the kth cluster of the observation
        X: np.ndarray
            2-D array of normalised data
        b: np.ndarray
            Calculated variational paramater B, derived from :func:`~vbvarsel.calcparams.calcB`
        a: np.ndarray
            Calculated variational paramater akj, derived from :func:`~vbvarsel.calcparams.calcAkj`
        m: np.ndarray
            Calculated variational paramater m, derived from :func:`~vbvarsel.calcparams.calcM`
        beta: np.ndarray
            Calculated variational paramater betakj, derived from :func:`~vbvarsel.calcparams.calcbetakj`
        d: int
            Shape parameter of the Beta distribution on the probability.
        C: np.ndarray
            Covariate selection indicators, :func:`~vbvarsel.calcparams.calcC`
        Z: np.ndarray
            Latent cluster assignment matrix
        sigma_0: np.ndarray
            N-dimensional array of squared sigma values
        mu_0: np.ndarray
            N-dimensional array of squared mu values
        T: float
            Annealing temperature, :func:`~vbvarsel.global_parameters.Hyperparameters.t_max`
        trick: bool (Optional) (Default: True)
            Flag for whether or not to use a mathematical trick to avoid numerical errors

    Returns
        C0: np.ndarray
            Calculated variational parameter C

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
