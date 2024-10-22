from scipy.special import digamma, psi, gammaln, gamma

import math
import numpy as np


# Evidence Lower Bound (ELBO) COMPUTATION
class ELBO_Computation:
    """
    A class to contain the Evidence Lower Bound (ELBO) computation.

    This class has no attributes, aside from Python's built-ins, and contains
    only methods, many of which are private.
    """

    def _ln_pi(self, alpha, k):
        """Private method to calculate Pi natural log.

        Params:
            alpha: calculated alphak value
            k: the Kth cluster

        Returns:
            Calculated ln(π)
        """
        return digamma(alpha[k]) - digamma(sum(alpha))

    def _ln_precision(self, a, b):
        """Private method for calculating precision natural log.

        Params:
            a: float
                calculated alphakj value
            b: np.ndarray
                calculated B value

        Returns:
            np.ndarray
                calculated precision natural log
        """
        if b > 1e-30:
            ld = np.log(b)
        else:
            ld = 0.0
        return psi(a) - ld

    def _log_resp_annealed(self, exp_ln_gam, exp_ln_mu, f0, N, K, C, T):
        """Private function to calculate log resp annealed.

        Params:
            exp_ln_gam: list
                expected ln(gamma) array
            exp_ln_mu: np.ndarray
                expected ln(mu) array
            f0: np.ndarray
                calculated f0 value
            N: int
                the Nth observation
            K: int
                number of clusters
            C: np.ndarray
                covariate selection indicator
            T: float
                annealing temperature

        Returns:
            log_resp: np.ndarray
                array of calculated values

        """
        log_resp = np.zeros((N, K))  # ln Z
        for k in range(K):
            log_resp[:, k] = (
                0.5 * exp_ln_gam[k]
                - 0.5 * sum(C) * np.log(2 * math.pi)
                - 0.5 * exp_ln_mu[:, k]
                + f0
            ) / T
        return log_resp

    def _ln_delta_annealed(self, C, j, d, T):
        """Private function to calculate the annealed value of delta natural log.

        Params:
            C: np.ndarray
                covariate selection indicator
            j: int
                iteration count
            d: np.ndarray
                Shape parameter of the Beta distribution on the probability.
            T: float
                annealing temperature

        Returns:
            ln_delta_ann: np.ndarray
                calculated array of natural log of annealed delta values
        """
        ln_delta_ann = digamma((C[j] + d + T - 1.0) / T) - digamma(
            (2 * d + 2 * T - 1.0) / T
        )
        return ln_delta_ann

    def _ln_delta_minus_annealed(self, C, j, d, T):
        """Private function to calculate the minus of annealed delta natural log.

        Params:
            C: np.ndarray
                covariate selection indicator
            j: int
                iteration count
            d: np.ndarray
                Shape parameter of the Beta distribution on the probability.
            T: float
                annealing temperature
        Returns:
            ln_delta_ann_minus: np.ndarray
                calculated array of natural log of annealed delta values

        """
        ln_delta_ann_minus = digamma((T - C[j] + d) / T) - digamma(
            (2 * d + 2 * T - 1.0) / T
        )
        return ln_delta_ann_minus

    def _entropy_wishart(self, k, j, b, a):
        """Private function to calculate wishart entropy.

        Params:
            K: int
                the kth cluster of the observation
            j: int
                iteration count
            b: np.ndarray
                calculated B value
            a: float
                calculated alphakj value
        Returns:
            e_w: np.ndarray
                array of entropy wishart values
        """
        if b[k][j, j] > 1e-30:
            ld = np.log(b[k][j, j])
        else:
            ld = 0.0
        e_w = gammaln(a[k][j]) - (a[k][j] - 1) * digamma(a[k][j]) - ld + a[k][j]
        return e_w

    def compute(
        self,
        XDim,
        K,
        N,
        C,
        Z,
        d,
        delta,
        beta,
        beta0,
        alpha,
        alpha0,
        a,
        a0,
        b,
        b0,
        m,
        m0,
        exp_ln_gam,
        exp_ln_mu,
        f0,
        T=1,
    ) -> float:
        """Function to compute the Evidence Lower Bound (ELBO). The ELBO is
        the useful lower-bound on the log-likelihood of observed data.

        Params:
            XDim: int
                number of columns
            K: int
                the kth cluster of the observation
            N: int
                the Nth observation
            C: np.ndarray
                covariate selection indicator
            Z: np.ndarray
                cluster assignment matrix
            d: np.ndarray
                Shape parameter of the Beta distribution on the probability.
            delta: int
                calculated delta value
            beta: np.ndarray
                calculated betakj value
            beta0: float
                Shrinkage parameter of the Gaussian conditional prior on the cluster means
            alpha: np.ndarray
                calculated alphak value
            alpha0: float
                Prior coefficient count
            a: float
                calculated alphakj value
            a0: np.ndarray
                Degrees of freedom for the Gamma prior on the cluster precision
            b: np.ndarray
                calculated B value
            b0: np.ndarray
                prior covariance
            m: np.ndarray
                calculated m value
            m0: np.ndarray
                prior mean
            exp_ln_gam: list
                expected ln(gamma) array
            exp_ln_mu: np.ndarray
                expected ln(mu) array
            f0: np.ndarray
                calculated f0 value
            T: float
                annealing temperature

        Returns:
            calculated ELBO
        """

        # nifty way to find out what the attributes are of the class I am using when
        # I am unfamiliar with how the mathematics of these functions works.

        # import inspect
        # import pprint
        # sig, elbo_locals = inspect.signature(self.compute), locals()
        # pprint.pprint([f"{str(_key)} : {type(elbo_locals[param.name])}" for _key, param in sig.parameters.items()])

        # # [<class 'int'>, <class 'int'>, <class 'int'>, <class 'numpy.ndarray'>,
        # # <class 'numpy.ndarray'>, <class 'int'>, <class 'numpy.ndarray'>,
        # # <class 'numpy.ndarray'>, <class 'float'>, <class 'numpy.ndarray'>,
        # # <class 'float'>, <class 'numpy.ndarray'>, <class 'float'>,
        # # <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'numpy.ndarray'>,
        # # <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'list'>,
        # # <class 'numpy.ndarray'>, <class 'numpy.ndarray'>, <class 'float'>]

        # E[ln p(X|Z, μ, Λ)]
        def _first_term(N, K, Z, C, exp_ln_gam, exp_ln_mu, f0, T):
            """Private internal function to calculate the 1st term of the ELBO

            Params:
                N: int
                    the nth observation
                K: int
                    the kth cluster of the observation
                Z: np.ndarray
                    cluster assignment matrix
                C: np.ndarray[int]
                    covariate selection indicator
                exp_ln_gam: list
                    expected ln(gamma) array
                exp_ln_mu: np.ndarray
                    expected ln(mu) array
                f0: np.ndarray
                    calculated f0 value
                T: float
                    Annealing temperature

            Returns:
                F2: float
                    first ELBO algorithm term
            """
            ln_resp = self._log_resp_annealed(exp_ln_gam, exp_ln_mu, f0, N, K, C, T)
            F2 = 0
            for n in range(N):
                for k in range(K):
                    F2 += Z[n, k] * (ln_resp[n, k])

            return F2

        # E[ln p(Z|π)]
        def _second_term(N, K, Z, alpha):
            """Private internal function to calculate ELBO 2nd term

            Params:
                N: int
                    the nth observation
                K: int
                    the kth cluster of the observation
                Z: np.ndarray
                    cluster assignment matrix
                alpha: np.ndarray
                    calculated alphak value
            Returns:
                s: float
                    calculated second ELBO algorithm term
            """
            s = 0
            for n in range(N):
                for k in range(K):
                    s += Z[n, k] * self._ln_pi(alpha, k)
            return s

        # E[ln p(π)]
        def _third_term(alpha0, K, alpha):
            """Private internal function to calculate the 3rd term of the ELBO

            Params:
                alpha0: float
                    Degrees of freedom for the Gamma prior on the cluster precision                K: int
                K: int
                    the kth cluster of the observation
                alpha: float
                    Prior coefficient count

            Return:
                a + b: float
                    calculated third term for ELBO algorithm

            """
            a = gammaln(alpha0 * K) - K * gammaln(alpha0)
            b = (alpha0 - 1) * sum([self._ln_pi(alpha, k) for k in range(K)])
            return a + b

        # E[ln p(μ, Λ)]
        def _fourth_term(K, XDim, beta0, beta, a0, a, b0, b, m, m0):
            """Private internal function to calculate the 4th term of the ELBO

            Params:
                K: int
                    the kth cluster of the observation
                XDim: int
                    number of columns
                beta0: float
                    Shrinkage parameter of the Gaussian conditional prior on the cluster
                beta: np.ndarray
                    calculated betakj values
                a0: int
                    Degrees of freedom for the Gamma prior on the cluster precision
                a: np.ndarray
                    calculated akj values
                b0: np.ndarray
                    prior covariance
                b: np.ndarray
                    calculated bkj value
                m: np.ndarray
                    calculated m value
                m0: np.ndarray
                    prior mean

            Returns:
                t: float
                    calculated fourth term of the ELBO algorithm
            """
            t = 0
            for k in range(K):
                for j in range(XDim):
                    F0 = 0.5 * np.log(beta0 / (2 * math.pi))
                    +0.5 * self._ln_precision(a[k, j], b[k][j, j])
                    if beta[k][j] > 0:
                        F1 = (
                            (beta0 * a[k][j] / beta[k][j]) * ((m[k][j] - m0[j]) ** 2)
                            + beta0 / beta[k][j]
                            + b0[j, j] * a[k][j] / beta[k][j]
                        )
                    else:
                        F1 = (
                            (beta0 * a[k][j]) * ((m[k][j] - m0[j]) ** 2)
                            + beta0
                            + b0[j, j] * a[k][j]
                        )

                    F2 = (
                        -np.log(gamma(a0))
                        + a0 * np.log(b0[j, j])
                        + (a0 - 2) * self._ln_precision(a[k, j], b[k][j, j])
                    )

                    t += F0 - F1 + F2

            return t

        # E[ln p(𝛾,𝛿)]
        def _fifth_term(XDim, d, C, T):
            """Private internal function to calculate the 5th term of the ELBO

            Params:
                XDim: int
                    number of columns
                d: np.ndarray
                    Shape parameter of the Beta distribution on the probability.
                C: np.ndarray[int]
                    covariate selection indicator matrix
                T: float
                    Annealing temperature

            Returns:
                a:
            """
            a = 0
            for j in range(XDim):
                F1 = (d + C[j] - 1) * self._ln_delta_annealed(C, j, d, T)
                F2 = (d + C[j]) * self._ln_delta_minus_annealed(C, j, d, T)
                F3 = np.log(gamma(2 * d)) - 2 * np.log(gamma(d))
                a += F1 + F2 + F3
            return a

        # E[ln q(Z)]
        def _sixth_term(Z:np.ndarray, N:int, K:int):
            """Private internal function to calculate the 6th term of the ELBO

            Params:
                Z: np.ndarray
                    cluster assignment matrix
                N: int
                    the Nth observation
                K: int
                    the Kth cluster of the observation

            Returns:
                a: ndarray
                    the calculated 6th term of the ELBO algorithm
            """
            a = 0
            for n in range(N):
                for k in range(K):
                    if Z[n, k] > 1e-30:
                        ld = np.log(Z[n, k])
                    else:
                        ld = 0.0
                    a += Z[n, k] * ld
            return a

        # E[ln q(π)]
        def _seventh_term(alpha:np.ndarray):
            """Private internal function to calculate the 7th term of the ELBO

            Params:
                alpha: np.ndarray
                    Calculated alpha value 

            Returns:
                a + b
            """
            a = sum([(alpha[k] - 1) * self._ln_pi(alpha, k) for k in range(K)])
            b = gammaln(sum(alpha)) - sum([gammaln(alpha[k]) for k in range(K)])
            return a + b

        # E[ln q(μ, Λ)]
        def _eighth_term(K, XDim, beta, a, b):
            """Private internal function to calculate the 8th term of the ELBO

            Params:
                K: int
                    the Kth cluster
                XDim: int
                    number of columns
                beta: ndarray[float]
                    claculated betakj values
                a: ndarray[float]
                    calculated alphakj values
                b: ndarray[float]
                    calculated bkj values

            Returns: 
                t: float
                    calculated 8th term for the ELBO algorithm
            """
            t = 0
            for k in range(K):
                for j in range(XDim):
                    t += (
                        0.5 * self._ln_precision(a[k, j], b[k][j, j])
                        + 0.5 * np.log(beta[k][j] / (2 * np.pi))
                        - 0.5
                        - self._entropy_wishart(k, j, b, a)
                    )
            return t

        # E[ln q(𝛾,𝛿)]
        def _ninth_term(XDim, d, delta, C, T):
            """Private internal function to calculate the 9th term of the ELBO

            Params:
                XDim: int
                    number of columns
                d: np.ndarray
                    Shape parameter of the Beta distribution on the probability.
                delta: int
                    calculated delta value
                C: np.ndarray[int]
                    covariate selection indicator matrix
                T: float
                    annealing temperature

            Returns:
                F0 + F1: float
                    calculated 9th term of the ELBO algorithm

            """
            F0 = (
                2
                * XDim
                * (
                    (d - 1) * (digamma(d) - digamma(2 * d))
                    + np.log(gamma(2 * d))
                    - 2 * np.log(gamma(d))
                )
            )
            F1 = 0
            for j in range(XDim):
                F1 += delta[j] * self._ln_delta_annealed(C, j, d, T) + (
                    1 - delta[j]
                ) * self._ln_delta_minus_annealed(C, j, d, T)

            return F0 + F1

        a_t = _first_term(N, K, Z, C, exp_ln_gam, exp_ln_mu, f0, T)
        b_t = _second_term(N, K, Z, alpha)
        c_t = _third_term(alpha0, K, alpha)
        d_t = _fourth_term(K, XDim, beta0, beta, a0, a, b0, b, m, m0)
        e_t = _fifth_term(XDim, d, C, T)
        f_t = _sixth_term(Z, N, K)
        g_t = _seventh_term(alpha)
        h_t = _eighth_term(K, XDim, beta, a, b)
        i_t = _ninth_term(XDim, d, delta, C, T)

        return a_t + b_t + c_t + d_t + e_t - (f_t + g_t + h_t + i_t) * T
