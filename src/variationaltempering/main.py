import numpy as np
import os
import pandas as pd

import hygiene
import dataSimCrook
from calcparams import *
from elbo import ELBO_Computation
from scipy.stats import beta
from sklearn.metrics import adjusted_rand_score


# classical geometric schedule T(k) = T0 * alpha^k
# T0 is the initial temperature
def geometric_schedule(T0, alpha, k, n):
    if k < n:
        return T0 * (alpha**k)
    else:
        return 1


# classical harmonic schedule T(k) = T0 / (1 + alpha * k)
# T0 is the initial temperature
def harmonic_schedule(T0, alpha, k):
    return T0 / (1 + alpha * k)


def establish_hyperparameters(**kwargs):
    '''Function to set hyperparameters, otherwise will return an object
    with default values.
    
    Paramaters:
        **kwargs: keywords arguments, see below.
    Returns:
        hyperparams: dict -> dictionary of hyperparameter values to be used
        in sumulations.

    Valid keyword arguments:
        `threshold`: a threshold for convergence, default is 1e-1
        `k1`: maximum clusters, default is 5
        `alpha0`: prior coefficient count, default is 1/k1
        `beta0`:
        `a0`:
        `d0`:
        `t_max`:
        `max_itr`:
        `max_models`:
    '''


    hyperparams = {
        #Convergence threshold
        "threshold" : kwargs.get('threshold', 1e-1),
        #Max clusters
        "k1" : kwargs.get('k1', 5),
        # shrinkage parameter of the Gaussian conditional prior on the cluster mean
        # influences the tightness and spread of the cluster, with smaller shrinkage leading to tighter clusters.
        "beta0" : kwargs.get("beta0", (1e-3)*1.),
        # degrees of freedom, for the Gamma prior on the cluster precision
        # controls the shape of the Gamma distribution, the higher the degree of freedom, the more peaked
        "a0" : kwargs.get("a0", 3.),
        # shape parameter of the Beta distribution on the probability of selection
        "d0" : kwargs.get('d0', 1),
        # maximum/starting annealing temperature
        # T_max = 1 means no annealing
        "t_max" : kwargs.get('t_max', 1.),
        # maximum number of iterations
        "max_itr" : kwargs.get("max_itr", 25),
        # number of models to run for model averaging
        "max_models" : kwargs.get("max_models", 10)
    }

    # prior coefficient count, or concentration parameter, for Dirichelet prior on the mixture proportions
    # this parameter can be interpreted as pseudocounts, i.e. the effective prior number of observations associated with each mixture component.
    #has to be calculated outside of the dict because it relies on another dict value and afaik there's no way to look introspectively into a dict
    hyperparams['alpha0'] = kwargs.get('alpha0', 1/hyperparams["k1"])

    return hyperparams

def establish_sim_params(n_observations:list[int]=[10], 
                         n_variables:int =50, 
                         n_relevants:list[int]=[10,50,80],
                         mixture_proportions:list[float]=[0.5, 0.3, 0.2],
                         means:list[int] = [0, 2, -2]):
    return {
        "n_observations":n_observations,
        "n_variables":n_variables,
        "n_relevants": n_relevants,
        "mixture_proportions": mixture_proportions,
        "means": means
    }        

# MAIN RUN FUNCTION
def run(
    X,
    m0,
    b0,
    C,
    hyperparameters,
    # simulated_vars,
    annealing="fixed",#
    max_annealed_itr=10,
    Ctrick=True,
    ):
    
    K = hyperparameters['k1']
    max_itr = hyperparameters['max_itr']
    threshold = hyperparameters['threshold']
    T = hyperparameters['t_max']
    alpha0 = hyperparameters['alpha0']
    beta0 = hyperparameters['beta0']
    a0 = hyperparameters['a0']
    b0 = hyperparameters['b0']
    d0 = hyperparameters['d0']
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
            cooling_rate = (1 / T) ** (1 / (max_annealed_itr - 1))
            T = geometric_schedule(T, cooling_rate, itr, max_annealed_itr)
        elif annealing == "harmonic":
            cooling_rate = (T - 1) / max_annealed_itr
            T = harmonic_schedule(T, cooling_rate, itr, max_annealed_itr)
        elif annealing == "fixed":
            T = T

        NK = Z.sum(axis=0)

        # M-like-step
        alphak = calcAlphak_annealed(NK=NK, alpha0=alpha0, T=T)
        akj = calcAkj_annealed(K=K, J=XDim, c=C, NK=NK, a0=a0, T=T)
        xd = calcXd(Z=Z, X=X)
        S = calcS(Z=Z, X=X, xd=xd)
        betakj = calcbetakj_annealed(K=K, XDim=XDim, c=C, NK=NK, beta0=beta0, T=T)
        m = calcM_annealed(
            K=K, XDim=XDim, beta0=beta0, m0=m0, NK=NK, xd=xd, betakj=betakj, c=C, T=T
        )
        bkj = calcB_annealed(
            W0=b0, xd=xd, K=K, m0=m0, XDim=XDim, beta0=beta0, S=S, c=C, NK=NK, T=T
        )
        delta = calcDelta_annealed(C=C, d=d0, T=T)

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
            d=d0,
            C=C,
            Z=Z,
            sigma_0=sigma_sq_0,
            mu_0=mu_0,
            T=T,
            trick=Ctrick,
        )

        lb = ELBO_Computation().compute(
            XDim=XDim,
            K=K,
            N=N,
            C=C,
            Z=Z,
            d=d0,
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

    return Z, lower_bound, C, itr


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

        normalised_data = hygiene.normalise_data(raw_data)
        shuffled_data = hygiene.shuffle_data(normalised_data)
        return shuffled_data
    return raw_data


if __name__ == "__main__":
    hp = establish_hyperparameters(t_max=11)
    print(hp)
    sp = establish_sim_params(
        n_observations = [10],
        n_variables = 200,
        n_relevants = [10,20,50,100,],
        mixture_proportions = [0.5, 0.3, 0.2],
        means = [0, 2, -2]
    )

    convergence_ELBO = []
    convergence_itr = []
    clust_predictions = []
    variable_selected = []
    times = []
    ARIs = []
    n_relevant_var = []
    n_obs = []

    n_rel_rel = []  # correct relevant

    n_irr_irr = []  # correct irrelevant

    for p, q in enumerate(sp["n_observations"]):
        for n, o in enumerate(sp["n_relevants"]):
            for i in range(hp["max_models"]):

                # print("Model " + str(i))
                # print("obs " + str(p))
                # print("rel " + str(n_rel))
                # print()

                n_relevant_var.append(sp["n_relevants"][n])
                # print(n_observations[p])
                n_obs.append(sp["n_observations"][p])

                variance_covariance_matrix = np.identity(sp["n_relevants"][n])
                # print(f"with loop: {np.shape(variance_covariance_matrix)}")
                data_crook = dataSimCrook.SimulateData(
                    sp["n_observations"][p],
                    sp["n_variables"],
                    sp["n_relevants"][n],
                    sp["mixture_proportions"],
                    sp["means"],
                    variance_covariance_matrix,
                )
                crook_data = data_crook.data_sim()
                perms = data_crook.permutation()
                data_crook.shuffle_sim_data(crook_data, perms)
                
                (N,XDim) = np.shape(crook_data)
        
            C = np.ones(XDim)  
            W0 = (1e-1)*np.eye(XDim) #prior cov (bigger: smaller covariance)
            m0 = np.zeros(XDim) #prior mean
            for j in range(XDim):
                m0[j] = np.mean(crook_data[:, j])
            
            delta0 = beta.rvs(1, 1, size=XDim)
            T = 1
            # Measure the execution time of the following code
            Z, lower_bound, Cs, iterations = run(X=data_crook.simulation_object["shuffled_data"],
                                     hyper_params = hp,
                                     delta0 = delta0,
                                     m0=m0, 
                                     b0=W0, 
                                     C=C)
            print(iterations)

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