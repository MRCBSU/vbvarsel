import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

from .dataSimCrook import SimulateCrookData
from .global_parameters import Hyperparameters, SimulationParameters
from .calcparams import * #pretty much anything that starts with calc is from here
from .elbo import ELBO_Computation
from .custodian import UserDataHandler

from sklearn.metrics import adjusted_rand_score

from dataclasses import dataclass, field

@dataclass(order=True)
class _Results:
    '''Dataclass to store the results of the simulation.'''
    convergence_ELBO: list[float] = field(default_factory=list)
    convergence_itr: list[int] = field(default_factory=list)
    # clust_predictions: list[int] = field(default_factory=list)
    # variable_selected: list[np.ndarray[float]] = field(default_factory=list)
    runtimes: list[float] = field(default_factory=list)
    ARIs: list[float] = field(default_factory=list)
    relevants: list[int] = field(default_factory=list)
    observations: list[int] = field(default_factory=list)
    correct_rel_vars: list[int] = field(default_factory=list)  # correct relevant
    correct_irr_vars: list[int] = field(default_factory=list)  # correct irrelevant
    
    def add_elbo(self, elbo:float) -> None:
        '''Function to append the ELBO convergence.'''
        self.convergence_ELBO.append(elbo)
    
    def add_convergence(self, iteration:int) -> None:
        '''Function to append convergence iteration.'''
        self.convergence_itr.append(iteration)

    # def add_prediction(self, predictions:list[int]) -> None:
    #     '''Function to append predicted cluster.'''
    #     self.clust_predictions.append(predictions)
    
    # def add_selected_variables(self, variables: np.ndarray[float]) -> None:
    #     '''Function to append selected variables.'''
    #     self.variable_selected.append(variables)

    def add_runtimes(self, runtime: float) -> None:
        '''Function to append runtime.'''
        self.runtimes.append(runtime)

    def add_ari(self, ari:float) -> None:
        '''Function to append the Adjusted Rand Index.'''
        self.ARIs.append(ari)
    
    def add_relevants(self, relevant: int) -> None:
        '''Function to append the relevant selected variables.'''
        self.relevants.append(relevant)

    def add_observations(self, observation: int) -> None:
        '''Function to append the number of observations.'''
        self.observations.append(observation)

    def add_correct_rel_vars(self, correct: int) -> None:
        '''Function to append the relevant correct variables.'''
        self.correct_rel_vars.append(correct)

    def add_correct_irr_vars(self, incorrect: int) -> None:
        '''Function to append the correct irrelevant variables.'''
        self.correct_irr_vars.append(incorrect)
    
    def save_results(self, path):
        savetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        savefile = f"results-{savetime}.csv"
        results_out = pd.DataFrame(self.__dict__)
        results_out.to_csv(path_or_buf=os.path.join(path, savefile), index=False)

# classical geometric schedule T(k) = T0 * alpha^k where k is the current iteration
# T0 is the initial temperature
def geometric_schedule(T0, alpha, itr, max_annealed_itr):
    '''Function to calculate geometric annealing.

    Params
    ------
        T0: int
            initial temperature for annealing.
        alpha: float
            cooling rate 
        itr: int
            current iteration
        max_annealed_itr: int
            maximum number of iteration to use annealing 

    Returns
    ------- 1, if itr >= max_annealed_itr, else T * alpha^itr
    '''
    if itr < max_annealed_itr:
        return T0 * (alpha**itr)
    else:
        return 1


# classical harmonic schedule T(k) = T0 / (1 + alpha * k) where k is the current iteration
# T0 is the initial temperature
def harmonic_schedule(T0, alpha, itr):
    '''Function to calculate harmonic annealing.

    Params
    ------
        T0: int
            the initial temperature
        alpha: float
            cooling rate 
        itr: int
            current iteration
    
    Returns
    -------
        Quotient of T0 by (1 + alpha * itr)
    '''
    return T0 / (1 + alpha * itr)


def _extract_els(el:int, unique_counts:np.ndarray, counts:np.ndarray) -> int:
    '''Function to extract elements from counts of a matrix.

    Params
    ------
        el: int 
            element of interest (can it only be 1 or 0?)
        unique_counts: NDarray
            array of unique counts from a matrix
        counts: NDarray
            array of total counts from a matrix
    Returns
    -------
        counts_of_element[0]:int -> integer of counts of targeted element.
    '''
    index_of_element = np.where(unique_counts == el)[0]
    counts_of_element = counts[index_of_element]
    if len(counts_of_element) == 0:
        return 0
    else:
        return counts_of_element[0]

# MAIN RUN FUNCTION
def _run_sim(
    X: np.ndarray[float],
    m0: np.ndarray[float],
    b0: np.ndarray[float],
    C: np.ndarray[float],
    hyperparameters: Hyperparameters,
    Ctrick:bool=True,
    annealing:str="fixed",
    max_annealed_itr:int=10,

    ) -> tuple:
    '''Private function to handle running the actual maths of the simulation. 
    Should not be called directly, it is used from the function `main`.


    X: np.ndarray[float]
        An array of shuffled and normalised data. Can be derived from a dataset
        the user has supplied or a simulated dataset from the `dataSimCrook`
        module. 
    m0: np.ndarray[float]
        2-D zeroed array
    b0: np.ndarray[float]
        2-D array with 1s in diagonal, zeroes in rest
    C: np.ndarray[int]
        covariate selection indicator
    CTrick: bool
       whether to use or not a mathematical trick to avoid numerical errors (Default: True)
    hyperparameters: Hyperparameters
        An object of specified hyperparameters
    annealing: str
        The type of annealing to apply to the simulation. Can be one of 
        "fixed", "geometric" or "harmonic", "fixed" does not apply annealing.
        (Default: "fixed")
    max_annealed_itr: int
        How many iterations to apply the annealing function. (Default: 10)


    Returns
    -------
    Tuple(Z: np.NDarray, lower_bound: list, _C: np.NDarray, itr: int)
        A tuple of experimental results.
        Z is an NDarray of Dirchilet data
        lower_bound is the calculated lower_bound of the experiment
        C is the calculated value of C
        itr is the number of iterations performed before convergence 
           
    '''
    K = hyperparameters.k1
    max_itr = hyperparameters.max_itr
    threshold = hyperparameters.threshold
    T = hyperparameters.t_max
    alpha0 = hyperparameters.alpha0
    beta0 = hyperparameters.beta0
    a0 = hyperparameters.a0
    d0 = hyperparameters.d0

    (N, XDim) = np.shape(X)

    # Params
    ------
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
        akj = calcAkj(K=K, J=XDim, C=C, NK=NK, a0=a0, T=T)
        xd = calcXd(Z=Z, X=X)
        S = calcS(Z=Z, X=X, xd=xd)
        betakj = calcbetakj(K=K, XDim=XDim, C=C, NK=NK, beta0=beta0, T=T)
        m = calcM(
            K=K, XDim=XDim, beta0=beta0, m0=m0, NK=NK, xd=xd, betakj=betakj, C=C, T=T
        )
        bkj = calcB(
            W0=b0, xd=xd, K=K, m0=m0, XDim=XDim, beta0=beta0, S=S, C=C, NK=NK, T=T
        )
        delta = calcDelta(C=C, d=d0, T=T)

        # E-like-step
        mu = expSigma(X=X, XDim=XDim, betak=betakj, m=m, b=bkj, a=akj, C=C)
        invc = expTau(b=bkj, a=akj, C=C)
        pik = expPi(alpha0=alpha0, NK=NK)
        f0 = calcF0(X=X, XDim=XDim, sigma_0=sigma_sq_0, mu_0=mu_0, C=C)

        Z = calcZ(
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
         Ctrick:bool = True,
         user_data:np.ndarray = None,
         annealing_type:str="fixed",
         save_output:bool=False):
    '''Function that runs the simulation.

    Params
    ------
        simulation_parameters: SimulationParameters
            An object of simulation paramaters to apply to the simulation. 
        hyperparameters: Hyperparameters
            An object of hyperparamters to apply to the simulation.
        Ctrick: bool
            Flag to determine whether or not to apply replica trick to the 
            simulation (Default: True)
        user_data: np.ndarray
            Optional parameter to allow users to supply their own data rather
            than working on a simulated dataset. User data MUST be in a 2-D 
            array, using the functions in the `custodian.py` module will reformat 
            and shuffle data to match the required format. (Default: None)
        annealing_type: str
            Optional type of annealing to apply to the simulation, can be one of
            "geometric", "harmonic" or "fixed", the latter of which does not
            apply any annealing. (Default: "fixed")
        save_output: bool
            Optional flag for users to save their output to a csv file. Data is
            saved in the current working directory with the file naming format
            "results-timestamp.csv". (Default: False)
    '''


    results = _Results() 

    test_data = UserDataHandler()
    test_data.load_data(user_data, normalise=True)
    perms = test_data.SimulatedValues.permutations
    
    for p, q in enumerate(simulation_parameters.n_observations):
        for n, o in enumerate(simulation_parameters.n_relevants):
            for i in range(hyperparameters.max_models):

                print("Model " + str(i))
                print("obs " + str(q))
                print("rel " + str(o))
                
                results.add_relevants(simulation_parameters.n_relevants[n])
                results.add_observations(simulation_parameters.n_observations[p])
                variance_covariance_matrix = np.identity(simulation_parameters.n_relevants[n])
                if user_data == None:
                    test_data = SimulateCrookData(
                        simulation_parameters.n_observations[p],
                        simulation_parameters.n_variables,
                        simulation_parameters.n_relevants[n],
                        simulation_parameters.mixture_proportions,
                        simulation_parameters.means,
                        variance_covariance_matrix,
                    )
                    crook_data = test_data.data_sim()
                    perms = test_data.permutation()
                    test_data.shuffle_sim_data(crook_data, perms)
                
                N, XDim = np.shape(test_data.SimulatedValues.data)
                C = np.ones(XDim)  
                W0 = (1e-1)*np.eye(XDim) #prior cov (bigger: smaller covariance)
                m0 = np.zeros(XDim) #prior mean
                for j in range(XDim):
                    m0[j] = np.mean(test_data.SimulatedValues.data[:, j])
                
                start_time = time.time()
                # Measure the execution time of the following code
                Z, lower_bound, Cs, iterations = _run_sim(
                    X=test_data.SimulatedValues.shuffled_data,
                    hyperparameters=hyperparameters,
                    m0=m0,
                    b0=W0,
                    C=C,
                    Ctrick=Ctrick,
                    annealing=annealing_type
                    )
                end_time = time.time()
                run_time = end_time - start_time
                print(f"runtime: {run_time}")
                results.add_runtimes(run_time)

                results.add_elbo(lower_bound[-1])
                results.add_convergence(iterations)
        
                clust_pred = [np.argmax(r) for r in Z]
                clust_pred = [int(x) for x in clust_pred]
        
                ari = adjusted_rand_score(np.array(test_data.SimulatedValues.true_labels),
                                          np.array(clust_pred))
                results.add_ari(ari)
        
                original_order = np.argsort(perms)
                var_selection_ordered = np.around(np.array(Cs)[original_order])
                
                #Find correct relevant variables
                unique_counts, counts = np.unique(
                    np.around(var_selection_ordered[:simulation_parameters.n_relevants[n]]),
                     return_counts=True
                     )
                # Extract the counts of the specific element from the counts array
                rel_counts_of_element = _extract_els(1, unique_counts, counts)
                results.add_correct_rel_vars(rel_counts_of_element)        

                #Find correct irrelevant variables
                unique_counts, counts = np.unique(
                    np.around(var_selection_ordered[simulation_parameters.n_relevants[n]:]),
                     return_counts=True
                     )

                # Extract the counts of the specific element from the counts array
                irr_counts_of_element = _extract_els(0, unique_counts, counts)
                results.add_correct_irr_vars(irr_counts_of_element)

    # print(results)
    # print(f"conv: {results.convergence_ELBO}")
    # print(f"iter: {results.convergence_itr}")
    # print(f"clusters: {results.clust_predictions}")
    # print(f"var sel: {results.variable_selected}")
    # print(f"time: {results.runtimes}")
    # print(f"aris: {results.ARIs}")
    # print(f"rels: {results.relevants}")
    # print(f"obs: {results.observations}")

    if save_output:
        results.save_results(os.getcwd())


# if __name__ == "__main__":

#     # simulationparameters = SimulationParameters([10,100], 50, [2,4,6], [0.2, 0.4, 0.7], [0,-2,2])
#     # print(simulationparameters.means[0])
#     import cProfile
#     import pstats

#     with cProfile.Profile() as profile:


    # hp = Hyperparameters(max_models=10)
    # sp = SimulationParameters(
    #     n_observations=[10, 100,1000],
    #     n_variables=200,
    #     n_relevants=[10, 20, 50, 100],
    #     mixture_proportions=[0.2, 0.3, 0.5])
    
    # main(sp, hp, save_output=True, Ctrick=True, user_data="test_input.csv")

#     pres = pstats.Stats(profile)
#     pres.sort_stats(pstats.SortKey.TIME)
#     pres.print_stats()
#     # # pass

#     # print(np.array(pd.read_csv("test.csv", header=None )[0]))

#     # df = UserDataHandler.load_data("BRCA.pam50.csv")
#     # UserDataHandler.normalise_data(df)
#     # print(UserDataHandler.SimulatedValues.data)