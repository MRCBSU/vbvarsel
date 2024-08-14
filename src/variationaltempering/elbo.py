from scipy.special import digamma, psi, gammaln, gamma

import math
import numpy as np


# Evidence Lower Bound (ELBO) COMPUTATION
class ELBO_Computation:
    '''
    A class to contain the Evidence Lower Bound (ELBO) computation.
    
    This class has no attributes, aside from Python's built-ins, and contains
    only methods, many of which are private.
    '''
    # Not sure if I should give this class Attributes
    # or not because quite frankly I don't know how all these forumulas interact
    # but I do like having the Class.method() syntax because I think it more
    # clearly indicates how it is used + all these functions were ONLY used in
    # the ELBO part of the original script, so they seem linked and therefore
    # Class-based logic makes sense.
    # def __init__(
    #     self,
    #     XDim,
    #     K,
    #     N,
    #     C,
    #     Z,
    #     d,
    #     delta,
    #     beta,
    #     beta0,
    #     alpha,
    #     alpha0,
    #     a,
    #     a0,
    #     b,
    #     b0,
    #     m,
    #     m0,
    #     exp_ln_pi,
    #     exp_ln_gam,
    #     exp_ln_mu,
    #     f0,
    #     T=1,
    # ):
    #     self.XDim = XDim
    #     self.K = K
    #     self.N = N
    #     self.C = C
    #     self.Z = Z
    #     self.d = d
    #     self.delta = delta
    #     self.beta = beta
    #     self.beta0 = beta0
    #     self.alpha = alpha
    #     self.alpha0 = alpha0
    #     self.a = a
    #     self.a0 = a0
    #     self.b = b
    #     self.b0 = b0
    #     self.m = m
    #     self.m0 = m0
    #     self.exp_ln_pi = exp_ln_pi
    #     self.exp_ln_gam = exp_ln_gam
    #     self.exp_ln_mu = exp_ln_mu
    #     self.f0 = f0
    #     self.T = T

    def _ln_pi(self, alpha, k):
        '''Private method to calculate Pi natural log.
        
        Params:
            alpha:
            k:

        Returns:
            difference of summed alpha's digamma and digamma of alpha[k]
        '''
        return digamma(alpha[k]) - digamma(sum(alpha))

    def _ln_precision(self, a, b):
        '''Private method for calculating precision natural log.

        Params:
            a:
            b:

        Returns:
            psi of a minus ld
        '''
        if b > 1e-30:
            ld = np.log(b)
        else:
            ld = 0.0
        return psi(a) - ld

    def _log_resp_annealed(self, exp_ln_gam, exp_ln_mu, f0, N, K, C, T):
        '''Private function to calculate log resp annealed.
        
        Params:
            exp_ln_gam:
            exp_ln_mu:
            f0:
            N:
            K:
            C:
            T:
        
        Returns:
            log_resp:
        
        '''
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
        '''Private function to calculate the annealed value of delta natural log.#
        
        Params:
            C:
            j:
            d:
            T:
        
        Returns:
            ln_delta_ann:
        '''
        ln_delta_ann = digamma((C[j] + d + T - 1.0) / T) - digamma((2 * d + 2 * T - 1.0) / T)
        return ln_delta_ann

    def _ln_delta_minus_annealed(self, C, j, d, T):
        '''Private function to calculate the minus of annealed delta natural log.

        Params:
            C:
            j:
            d:
            T:
        Returns:
            ln_delta_ann_minus:    
        
        '''
        ln_delta_ann_minus = digamma((T - C[j] + d) / T) - digamma((2 * d + 2 * T - 1.0) / T)
        return ln_delta_ann_minus
        
    def _entropy_wishart(self, k, j, b, a):
        '''Private function to calculate wishart entropy.

        Params:
            k:
            j:
            b:
            a:
        Returns:
            e_w:
        '''
        if b[k][j, j] > 1e-30:
            ld = np.log(b[k][j, j])
        else:
            ld = 0.0
        e_w =  gammaln(a[k][j]) - (a[k][j] - 1) * digamma(a[k][j]) - ld + a[k][j]
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
        exp_ln_pi,
        exp_ln_gam,
        exp_ln_mu,
        f0,
        T=1,
    ):
        
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
        def _first_term(N, K, Z, C, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T):
            ln_resp = self._log_resp_annealed(exp_ln_gam, exp_ln_mu, f0, N, K, C, T)
            F2 = 0
            for n in range(N):
                for k in range(K):
                    F2 += Z[n, k] * (ln_resp[n, k])

            return F2

        # E[ln p(Z|π)]
        def _second_term(N, K, Z, alpha):
            s = 0
            for n in range(N):
                for k in range(K):
                    s += Z[n, k] * self._ln_pi(alpha, k)
            return s

        # E[ln p(π)]
        def _third_term(alpha0, K, alpha):
            a = gammaln(alpha0 * K) - K * gammaln(alpha0)
            b = (alpha0 - 1) * sum([self._ln_pi(alpha, k) for k in range(K)])
            return a + b

        # E[ln p(μ, Λ)]
        def _fourth_term(K, XDim, beta0, beta, a0, a, b0, b, m, m0):
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
            a = 0
            for j in range(XDim):
                F1 = (d + C[j] - 1) * self._ln_delta_annealed(C, j, d, T)
                F2 = (d + C[j]) * self._ln_delta_minus_annealed(C, j, d, T)
                F3 = np.log(gamma(2 * d)) - 2 * np.log(gamma(d))
                a += F1 + F2 + F3
            return a

        # E[ln q(Z)]
        def _sixth_term(Z, N, K):
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
        def _seventh_term(alpha):
            a = sum([(alpha[k] - 1) * self._ln_pi(alpha, k) for k in range(K)])
            b = gammaln(sum(alpha)) - sum([gammaln(alpha[k]) for k in range(K)])
            return a + b

        # E[ln q(μ, Λ)]
        def _eighth_term(K, XDim, beta, a, b):
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

        a_t = _first_term(N, K, Z, C, exp_ln_pi, exp_ln_gam, exp_ln_mu, f0, T)
        b_t = _second_term(N, K, Z, alpha)
        c_t = _third_term(alpha0, K, alpha)
        d_t = _fourth_term(K, XDim, beta0, beta, a0, a, b0, b, m, m0)
        e_t = _fifth_term(XDim, d, C, T)
        f_t = _sixth_term(Z, N, K)
        g_t = _seventh_term(alpha)
        h_t = _eighth_term(K, XDim, beta, a, b)
        i_t = _ninth_term(XDim, d, delta, C, T)

        return a_t + b_t + c_t + d_t + e_t - (f_t + g_t + h_t + i_t) * T
