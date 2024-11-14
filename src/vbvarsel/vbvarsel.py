import numpy as np
import pandas as pd
import time
import os
from datetime import datetime

from .data_sim_crook import SimulateCrookData
from .global_parameters import Hyperparameters, SimulationParameters
from .calcparams import * #pretty much anything that starts with "calc" is from here
from .elbo import ELBO_Computation
from .custodian import UserDataHandler

from sklearn.metrics import adjusted_rand_score

from dataclasses import dataclass, field

@dataclass(order=True)
class _Results:
    '''Dataclass object to store clustering results.'''
    convergence_ELBO: list[float] = field(default_factory=list)
    convergence_itr: list[int] = field(default_factory=list)
    clust_predictions: list[int] = field(default_factory=list)
    variable_selected: list[np.ndarray[float]] = field(default_factory=list)
    runtimes: list[float] = field(default_factory=list)
    ARIs: list[float] = field(default_factory=list)
    relevants: list[int] = field(default_factory=list)
    observations: list[int] = field(default_factory=list)
    correct_rel_vars: list[int] = field(default_factory=list)  # correct relevant
    correct_irr_vars: list[int] = field(default_factory=list)  # correct irrelevant
    
    def add_elbo(self, elbo:float) -> None:
        '''Method to append the ELBO convergence.'''
        self.convergence_ELBO.append(elbo)
    
    def add_convergence(self, iteration:int) -> None:
        '''Method to append convergence iteration.'''
        self.convergence_itr.append(iteration)

    def add_prediction(self, predictions:list[int]) -> None:
        '''Method to append predicted cluster.'''
        self.clust_predictions.append(predictions)
    
    def add_selected_variables(self, variables: np.ndarray[float]) -> None:
        '''Method to append selected variables.'''
        self.variable_selected.append(variables)

    def add_runtimes(self, runtime: float) -> None:
        '''Method to append runtime.'''
        self.runtimes.append(runtime)

    def add_ari(self, ari:float) -> None:
        '''Method to append the Adjusted Rand Index.'''
        self.ARIs.append(ari)
    
    def add_relevants(self, relevant: int) -> None:
        '''Method to append the relevant selected variables.'''
        self.relevants.append(relevant)

    def add_observations(self, observation: int) -> None:
        '''Method to append the number of observations.'''
        self.observations.append(observation)

    def add_correct_rel_vars(self, correct: int) -> None:
        '''Method to append the relevant correct variables.'''
        self.correct_rel_vars.append(correct)

    def add_correct_irr_vars(self, incorrect: int) -> None:
        '''Method to append the correct irrelevant variables.'''
        self.correct_irr_vars.append(incorrect)
    
    def save_results(self):
        '''Method to save results to a csv, using datetime format for naming.'''
        path = os.getcwd()
        savetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        savefile = f"results-{savetime}.csv"
        results_out = pd.DataFrame(self.__dict__)
        results_out.to_csv(path_or_buf=os.path.join(path, savefile), index=False)

# classical geometric schedule T(k) = T * alpha^k where k is the current iteration
# T is the initial temperature
def geometric_schedule(T: int, alpha: float, itr: int, max_annealed_itr: int) -> float:
    '''Function to calculate geometric annealing.

    Params
        T: int
            initial temperature for annealing.
        alpha: float
            cooling rate 
        itr: int
            current iteration
        max_annealed_itr: int
            maximum number of iteration to use annealing 
    Returns
        float:
            1, if itr >= max_annealed_itr, else T0 * alpha^itr

    '''
    if itr < max_annealed_itr:
        return T * (alpha**itr)
    else:
        return 1


# classical harmonic schedule T(k) = T0 / (1 + alpha * k) where k is the current iteration
# T0 is the initial temperature
def harmonic_schedule(T: int, alpha: float, itr: int) -> float:
    '''Function to calculate harmonic annealing.

    Params
        T: int
            the initial temperature
        alpha: float
            cooling rate 
        itr: int
            current iteration
    Returns
        float:
            Quotient of T by (1 + alpha * itr)

    '''
    return T / (1 + alpha * itr)


def _extract_els(el:int, unique_counts:np.ndarray, counts:np.ndarray) -> int:
    '''Function to extract elements from counts of a matrix.

    Params
        el: int 
            element of interest (can it only be 1 or 0?)
        unique_counts: NDarray
            array of unique counts from a matrix
        counts: NDarray
            array of total counts from a matrix
    Returns
        int: integer of counts of targeted element.

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
    ) -> tuple:
    '''Private function to handle running the actual maths of the simulation. 
    Should not be called directly, it is used from the function `main()`.

    Params
        X: np.ndarray[float]
            An array of shuffled and normalised data. Can be derived from a dataset
            the user has supplied or a simulated dataset from the `dataSimCrook`
            module. 
        m0: np.ndarray[float]
            2-D zeroed array
        b0: np.ndarray[float]
            2-D array with 1s in diagonal, zeroes in rest
        C: np.ndarray[int]
            covariate selection indicators
        hyperparameters: Hyperparameters
            An object of specified hyperparameters
        CTrick: bool (Optional) (Default: True)
            whether to use or not a mathematical trick to avoid numerical errors 
        annealing: str (Optional) (Default: "fixed")
            The type of annealing to apply to the simulation. Can be one of 
            "fixed", "geometric" or "harmonic", "fixed" does not apply annealing.
            

    Returns
        Tuple:
            Z: np.ndarray[float]
                an NDarray of Dirchilet data
            lower_bound: list[float]
                List of the calculated estimated lower bounds of the experiment
            C: np.ndarray[float]
                Calculated covariate selection indicators. 
            itr: int
                is the number of iterations performed before convergence

    '''
    K = hyperparameters.k1
    max_itr = hyperparameters.max_itr
    threshold = hyperparameters.threshold
    T = hyperparameters.t_max
    alpha0 = hyperparameters.alpha0
    beta0 = hyperparameters.beta0
    a0 = hyperparameters.a0
    d0 = hyperparameters.d0
    max_annealed_itr = hyperparameters.max_annealed_itr

    (N, XDim) = np.shape(X)

    # Params

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
        esig = expSigma(X=X, XDim=XDim, betak=betakj, m=m, b=bkj, a=akj, C=C)
        invc = expTau(bkj=bkj, akj=akj, C=C)
        pik = expPi(alpha0=alpha0, NK=NK)
        f0 = calcF0(X=X, XDim=XDim, sigma_0=sigma_sq_0, mu_0=mu_0, C=C)

        Z = calcZ(
            exp_ln_pi=pik, exp_ln_tau=invc, exp_ln_sigma=esig, f0=f0, N=N, K=K, C=C, T=T
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
            exp_ln_tau=invc,
            exp_ln_sigma=esig,
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




def main(
         hyperparameters: Hyperparameters,
         simulation_parameters: SimulationParameters, 
         Ctrick:bool = True,
         user_data: str | os.PathLike = None,
         user_labels: str | list[str] = None,
         cols_to_skip: list[str] = None,
         annealing_type:str="fixed",
         save_output:bool=False) -> _Results:
    '''The main entry point to the package.

    Params
    
        hyperparameters: Hyperparameters (Required)
            An object of hyperparamters to apply to the simulation.
        simulation_parameters: SimulationParameters (Optional) (Default: `SimulationParameters()`)
            An object of simulation paramaters to apply to the simulation.
            Note: This is a required parameter if a user does not supply
            their own data. 
        Ctrick: bool (Optional) (Default: True)
            Flag to determine whether or not to apply replica trick to the 
            simulation 
        user_data: str or os.PathLike (Optional) (Default: None)
            A location of a csv document for data a user whishes to test.
        user_labels: str | list[str] (Optional) (Default: None)
            A string or list of strings to identify labels. A string value will
            try to extract a column of the same name from the supplied data.
        cols_to_skip: list[str] (Optional) (Default: None)
            An optional list of columns to drop from the dataframe. This should
            be used to remove any non-numeric data from the dataframe. If a 
            column shares the same name as a label column, the labels will be
            extracted before the column is dropped. 

            **Hint**: an unnamed column can be passed by using "Unnamed: [index]",
            eg "Unnamed: 0" to drop a blank name first column.
        annealing_type: str (Optional) (Default: "fixed")
            Optional type of annealing to apply to the simulation, can be one of
            "geometric", "harmonic" or "fixed", the latter of which does not
            apply any annealing. 
        save_output: bool (Optional) (Default: False)
            Optional flag for users to save their output to a csv file. Data is
            saved in the current working directory with the file naming format
            "results-timestamp.csv". 
    Returns
    
        results: dataclass
            An object of results stored in a series of arrays from the clustering
            algorithm. Some arrays may be populated by `nan` values. This is the
            case if a user supplies their own data but does not have corresponding
            labels. Additionally, some fields are only captured during entirely
            simulated runs, as such will be `nan`-ed if a user provides their own
            dataset.

    '''

    results = _Results() 

    if user_data:
        test_data = UserDataHandler()
        test_data.load_data(data_source=user_data, cols_to_ignore=cols_to_skip, labels=user_labels)
        simulation_parameters.n_observations = [test_data.ExperimentValues.data.shape[0]]
        simulation_parameters.n_relevants = [test_data.ExperimentValues.data.shape[0]]
        perms = test_data.ExperimentValues.permutations
    #instantiate user data outside of the loop, because most of the loop is for creating the simulated data.

    ####BEGIN SIMULATION ONLY
    #IF USER DATA IGNORE THE FIRST TWO LOOPS
    #FOR USER DATA RUN ONLY MAX MODELS AMOUNT OF TIMES
    print(simulation_parameters)
    print(hyperparameters)
    for p, q in enumerate(simulation_parameters.n_observations): #nrows of user data [100]
        for n, o in enumerate(simulation_parameters.n_relevants): #nrows or anything [100]
            for i in range(hyperparameters.max_models):
                
                #COMMENT/DELETE after
                print("Model " + str(i))
                print("obs " + str(q))
                print("rel " + str(o))
                
                if user_data == None:
                    results.add_relevants(simulation_parameters.n_relevants[n])
                    results.add_observations(simulation_parameters.n_observations[p])
                    variance_covariance_matrix = np.identity(simulation_parameters.n_relevants[n])
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

                
                ##THIS APPLIES TO EVERYTHING (SIMULATED AND NON-SIMULATED)
                N, XDim = np.shape(test_data.ExperimentValues.data)
                C = np.ones(XDim)  
                W0 = (1e-1)*np.eye(XDim) #prior cov (bigger: smaller covariance)
                m0 = np.zeros(XDim) #prior mean
                for j in range(XDim):
                    m0[j] = np.mean(test_data.ExperimentValues.data[:, j])
                
                start_time = time.time()
                # Measure the execution time of the following code
                Z, lower_bound, Cs, iterations = _run_sim(
                    X=test_data.ExperimentValues.shuffled_data,
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
                results.add_prediction(clust_pred)
            
                #only with true labels
                #expected value for pam50 ~0.5 or so
                if ((user_labels is not None) and (len(user_labels) > 0)) or (not user_data):
                    ari = adjusted_rand_score(np.array(test_data.ExperimentValues.true_labels),
                                            np.array(clust_pred))
                    results.add_ari(ari)
                else:
                    results.add_ari(np.nan)
        
                original_order = np.argsort(perms)
                #ADD THIS TO THE RESULTS OBJECT
                var_selection_ordered = np.around(np.array(Cs)[original_order])
                results.add_selected_variables(var_selection_ordered)
                ###TO END ONLY FOR SIMULATION
                #Find correct relevant variables
                if user_data == None:
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
                else:
                    #because theres no values, the arrays arent the same length and it cant be saved
                    #so this is kind of hacky but it works
                    results.add_correct_irr_vars(np.nan)
                    results.add_correct_rel_vars(np.nan)
                    results.add_relevants(np.nan)
                    results.add_observations(np.nan)


                #USERS SHOULD GET CSV WITH RUNTIME, CONVERGENCE, ELBO, ARI (if labels), VAR_SELECTION_ORDERED AND CLUST PREDICTIONS
                # print(results)
                print(f"conv: {results.convergence_ELBO}")
                print(f"iter: {results.convergence_itr}")
                print(f"clusters: {results.clust_predictions}")
                print(f"var sel: {results.variable_selected}")
                print(f"time: {results.runtimes}")
                print(f"aris: {results.ARIs}")
                print(f"rels: {results.relevants}")
                print(f"obs: {results.observations}")
    if save_output:
        results.save_results()

    #still want to return the results because this will probably just be 1 step in a series of steps
    return results