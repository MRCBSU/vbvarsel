from calcparams import *
from elbo import *
import numpy as np
import typing
import typing_extensions
import os
import pandas as pd


# MAIN RUN FUNCTION
def run(
    X,
    K,
    alpha0,
    beta0,
    v0,
    a0,
    m0,
    b0,
    d,
    delta0,
    C,
    threshold,
    max_itr,
    T0=1.0,
    annealing="fixed",
    max_annealed_itr=10,
    Ctrick=True,
):
    (N, XDim) = np.shape(X)

    # params:
    Z = np.array([np.random.dirichlet(np.ones(K)) for _ in range(N)])

    # parameter estimates for \Phi_{0j} precomputed as MLE
    mu_0 = np.zeros(XDim)
    sigma_sq_0 = np.ones(XDim)
    for j in range(XDim):
        mu_0[j] = sum(X[:, j]) / N
        sigma_sq_0[j] = sum((X[:, j] - mu_0[j]) ** 2) / N

    itr = 0

    lower_bound = []
    converged = False

    while itr < max_itr:

        if annealing == "geometric":
            cooling_rate = (1 / T0) ** (1 / (max_annealed_itr - 1))
            T = geometric_schedule(T0, cooling_rate, itr, max_annealed_itr)
        elif annealing == "harmonic":
            cooling_rate = (T0 - 1) / max_annealed_itr
            T = harmonic_schedule(T0, cooling_rate, itr, max_annealed_itr)
        elif annealing == "fixed":
            T = T0

        NK = Z.sum(axis=0)

        # M-like-step

        alphak = calcAlphak_annealed(NK=NK, alpha0=alpha0, T=T)
        akj = calcAkj_annealed(K=K, J=XDim, c=C, NK=NK, a0=a0, T=T)
        xd = calcXd(Z=Z, X=X)
        S = calcS(Z=Z, X=X, xd=xd)
        betakj = calcbetakj_annealed(K=K, XDim=XDim, c=C, NK=NK, beta0=beta0, T=T)
        m = calcM_annealed(
            K=K, XDim=XDim, beta0=beta0, m0=m0, NK=NK, xd=xd, S=betakj, c=C, T=T
        )
        bkj = calcB_annealed(
            W0=b0, xd=xd, K=K, m0=m0, XDim=XDim, beta0=beta0, S=S, c=C, NK=NK, T=T
        )
        delta = calcDelta_annealed(C=C, d=d, T=T)

        # E-like-step
        mu = expSigma(X=X, XDim=XDim, betak=betakj, m=m, b=bkj, a=akj, C=C)
        invc = expTau(b=bkj, a=akj, C=C)
        pik = expPi(alpha0=alpha0, NK=NK)
        f0 = calcF0(X=X, XDim=XDim, sigma_0=sigma_sq_0, mu_0=mu_0, C=C)

        Z = calcZ_annealed(
            exp_ln_pi=pik, exp_ln_gam=invc, exp_ln_mu=mu, f0=f0, N=N, K=K, C=C, T=T
        )
        C = calcC_annealed(
            XDim=XDim,
            N=N,
            K=K,
            X=X,
            b=bkj,
            a=akj,
            m=m,
            beta=betakj,
            d=d,
            C=C,
            Z=Z,
            sigma_0=sigma_sq_0,
            mu_0=mu_0,
            T=T,
            trick=Ctrick,
        )

        lb = elbo_computation(
            XDim=XDim,
            K=K,
            N=N,
            C=C,
            Z=Z,
            d=d,
            delta=delta,
            beta=betakj,
            beta0=beta0,
            alpha=alphak,
            alpha0=alpha0,
            a=akj,
            a0=a0,
            b=bkj,
            b0=b0,
            m=m,
            m0=m0,
            exp_ln_pi=pik,
            exp_ln_gam=invc,
            exp_ln_mu=mu,
            f0=f0,
            T=T,
        )
        lower_bound.append(lb)

        # Convergence criterion
        improve = (lb - lower_bound[itr - 1]) if itr > 0 else lb
        if itr > 0 and 0 < improve < threshold:
            print("Converged at iteration {}".format(itr))
            converged = True
            break

        itr += 1

    return m, S, invc, pik, Z, lower_bound, C, itr


def load_data(data_loc: str | os.PathLike, clean_too: bool = False) -> list:
    """Loads data to be be used in simulations with option to clean data.

    Parameters:

    data_loc: str|os.Pathlike
        The file location of the spreadsheet, must be in CSV format.
    clean_too: bool, optional
        A flag to enable pre-determined cleaning. Should only be set to TRUE if
        using PAM50 datasets, data reformatting operations may not apply to
        other datasets, in which case a user should take the returned dataframe
        from this function and reformat their data as needed.

    Returns:

    raw_data|shuffled_data: list
        An array of data.
    """

    raw_data = pd.DataFrame(data_loc)

    if clean_too:
        import hygiene

        normalised_data = hygiene.normalise_data(raw_data)
        shuffled_data = hygiene.shuffle_data(normalised_data)
        return shuffled_data
    return raw_data

if __name__ == '__main__':
    pass