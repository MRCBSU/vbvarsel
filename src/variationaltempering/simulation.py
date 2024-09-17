import numpy as np
import pandas as pd
import time

from dataSimCrook import SimulateCrookData
from global_parameters import Hyperparameters, SimulationParameters
from calcparams import *
from elbo import ELBO_Computation

from scipy.stats import beta
from sklearn.metrics import adjusted_rand_score

from dataclasses import dataclass, field

# classical geometric schedule T(k) = T0 * alpha^k where k is the current iteration
# T0 is the initial temperature
def geometric_schedule(T0, alpha, itr, max_annealed_itr):
    '''Function to calculate geometric annealing.

    Params:
        T0: int -> initial temperature for annealing.
        alpha: float -> cooling rate 
        itr: int -> current iteration
        max_annealed_itr: int -> maximum number of iteration to use annealing 

    Returns: 1, if itr >= max_annealed_itr, else T * alpha^itr
    '''
    if itr < max_annealed_itr:
        return T0 * (alpha**itr)
    else:
        return 1


# classical harmonic schedule T(k) = T0 / (1 + alpha * k) where k is the current iteration
# T0 is the initial temperature
def harmonic_schedule(T0, alpha, itr):
    '''Function to calculate harmonic annealing.

    Params:
        T0: int -> the initial temperature
        alpha: float -> cooling rate 
        itr: int -> current iteration
    
    Returns:
        Quotient of T0 by (1 + alpha * itr)
    '''
    return T0 / (1 + alpha * itr)


def extract_els(el:int, unique_counts:np.ndarray, counts:np.ndarray) -> int:
    '''Function to extract elements from counts of a matrix.

    Params:
        el:int -> element of interest (can it only be 1 or 0?)
        unique_counts:ndarray -> array of unique counts from a matrix
        counts:ndarray -> array of total counts from a matrix
    Returns:
        counts_of_element[0]:int -> integer of counts of targeted element.
    '''
    index_of_element = np.where(unique_counts == el)[0]
    counts_of_element = counts[index_of_element]
    if len(counts_of_element) == 0:
        return 0
    else:
        return counts_of_element[0]

# MAIN RUN FUNCTION
def run_sim(
    X,
    m0,
    b0,
    C,
    hyperparameters: Hyperparameters,
    annealing="fixed",
    max_annealed_itr=10,
    Ctrick=True,
    ):
    
    K = hyperparameters.k1
    max_itr = hyperparameters.max_itr
    threshold = hyperparameters.threshold
    T = hyperparameters.t_max
    alpha0 = hyperparameters.alpha0
    beta0 = hyperparameters.beta0
    a0 = hyperparameters.a0
    d0 = hyperparameters.d0

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
            T = harmonic_schedule(T, cooling_rate, itr)
        elif annealing == "fixed":
            T = T

        NK = Z.sum(axis=0)

        # M-like-step
        alphak = calcAlphak(NK=NK, alpha0=alpha0, T=T)
        akj = calcAkj(K=K, J=XDim, c=C, NK=NK, a0=a0, T=T)
        xd = calcXd(Z=Z, X=X)
        S = calcS(Z=Z, X=X, xd=xd)
        betakj = calcbetakj(K=K, XDim=XDim, c=C, NK=NK, beta0=beta0, T=T)
        m = calcM(
            K=K, XDim=XDim, beta0=beta0, m0=m0, NK=NK, xd=xd, betakj=betakj, c=C, T=T
        )
        bkj = calcB(
            W0=b0, xd=xd, K=K, m0=m0, XDim=XDim, beta0=beta0, S=S, c=C, NK=NK, T=T
        )
        delta = calcDelta(C=C, d=d0, T=T)

        # E-like-step
        mu = expSigma(X=X, XDim=XDim, betak=betakj, m=m, b=bkj, a=akj, C=C)
        invc = expTau(b=bkj, a=akj, C=C)
        pik = expPi(alpha0=alpha0, NK=NK)
        f0 = calcF0(X=X, XDim=XDim, sigma_0=sigma_sq_0, mu_0=mu_0, C=C)

        Z0 = calcZ(
            exp_ln_pi=pik, exp_ln_gam=invc, exp_ln_mu=mu, f0=f0, N=N, K=K, C=C, T=T
        )
        C = calcC(
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
            Z=Z0,
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
            Z=Z0,
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




def main(simulation_parameters: SimulationParameters, 
         hyperparameters: Hyperparameters, 
         annealing_type:str="fixed"):
    '''Function that runs the simulation.

    Params:
        sp: object -> An object of simulation paramaters to apply to the simulation. 
            For more information please see the `establish_sim_params` function.
        hp: object -> An object of hyperparamters to apply to the simulation.
            For more information please see the `establish_hyperparameters` function.
    '''
    convergence_ELBO = []
    convergence_itr = []
    clust_predictions = []
    variable_selected = []
    times = []
    ARIs = []
    n_relevant_var = []
    n_obs = []

    n_rel_correct = []  # correct relevant

    n_irr_correct = []  # correct irrelevant

    for p, q in enumerate(simulation_parameters.n_observations):
        for n, o in enumerate(simulation_parameters.n_relevants):
            for i in range(hyperparameters.max_models):

                print("Model " + str(i))
                print("obs " + str(q))
                print("rel " + str(o))
                # print()

                n_relevant_var.append(simulation_parameters.n_relevants[n])
                # print(n_observations[p])
                n_obs.append(simulation_parameters.n_observations[p])

                variance_covariance_matrix = np.identity(simulation_parameters.n_relevants[n])
                # print(f"with loop: {np.shape(variance_covariance_matrix)}")
                data_crook = SimulateCrookData(
                    simulation_parameters.n_observations[p],
                    simulation_parameters.n_variables,
                    simulation_parameters.n_relevants[n],
                    simulation_parameters.mixture_proportions,
                    simulation_parameters.means,
                    variance_covariance_matrix,
                )
                crook_data = data_crook.data_sim()
                perms = data_crook.permutation()
                data_crook.shuffle_sim_data(crook_data, perms)
                
                N, XDim = np.shape(crook_data)
                C = np.ones(XDim)  
                W0 = (1e-1)*np.eye(XDim) #prior cov (bigger: smaller covariance)
                m0 = np.zeros(XDim) #prior mean
                for j in range(XDim):
                    m0[j] = np.mean(crook_data[:, j])
                
                # delta0 = beta.rvs(1, 1, size=XDim)
                start_time = time.time()
                # Measure the execution time of the following code
                Z, lower_bound, Cs, iterations = run_sim(X=data_crook.simulation_object["shuffled_data"],
                                        hyperparameters=hyperparameters,
                                        m0=m0, 
                                        b0=W0, 
                                        C=C,
                                        annealing=annealing_type)
                end_time = time.time()
                run_time = end_time - start_time
                print(f"runtime: {run_time}")
                times.append(run_time)

                convergence_ELBO.append(lower_bound[-1])
                convergence_itr.append(iterations)
        
                clust_pred = [np.argmax(r) for r in Z]
                clust_predictions.append(clust_pred)
        
                ari = adjusted_rand_score(np.array(data_crook.simulation_object['true_labels']), np.array(clust_pred))
                ARIs.append(ari)
        
                original_order = np.argsort(perms)
                var_selection_ordered = np.around(np.array(Cs)[original_order])
                variable_selected.append(var_selection_ordered)
                
                #Find correct relevant variables
                unique_counts, counts = np.unique(np.around(var_selection_ordered[:simulation_parameters.n_relevants[n]]), return_counts=True)
                # Find the index of the specific element (e.g., element 0) in the unique_counts array
                element_to_extract = 1
                # Extract the counts of the specific element from the counts array
                counts_of_element = extract_els(element_to_extract, unique_counts, counts)
                n_rel_correct.append(counts_of_element)        

                #Find correct irrelevant variables
                unique_counts, counts = np.unique(np.around(var_selection_ordered[simulation_parameters.n_relevants[n]:]), return_counts=True)
                # Find the index of the specific element (e.g., element 0) in the unique_counts array
                element_to_extract = 0
                # Extract the counts of the specific element from the counts array
                counts_of_element = extract_els(element_to_extract, unique_counts, counts)
                n_irr_correct.append(counts_of_element)

    print(f"conv: {convergence_ELBO} \n, inter: {convergence_itr} \n, clusters: {clust_predictions} \n \
                var sel: {variable_selected} \n time: {times} \n  aris: {ARIs} \n rels: {n_relevant_var} \n  obs: {n_obs} \n")

if __name__ == "__main__":

    simulationparameters = SimulationParameters([10,100], 50, [2,4,6], [0.2, 0.4, 0.7], [0,-2,2])
    print(simulationparameters)

    # hp = establish_hyperparameters()
    # sp = establish_sim_params()

    # main(sp, hp, annealing_type="geometric")
    # pass