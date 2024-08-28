# import sys
# sys.path.append(r"C:\Users\Alan\Desktop\dev\variationalTempering_beta\src")
from variationaltempering.simulation import establish_hyperparameters, establish_sim_params

def test_default_hyperparameters():
    default_hypers = establish_hyperparameters()
    assert default_hypers == {
        "threshold" : 1e-1,
        "k1" :  5,
        "beta0" : 1e-3,
        "alpha0" : .2,
        "a0" :  3.,
        "d0" :  1,
        "t_max" : 1.,
        "max_itr" : 25,
        "max_models" : 10
    }

def test_establish_sim_params():
    default_sim_params = establish_sim_params()
    assert default_sim_params == {
        "n_observations": [100, 1000],
        "n_variables": 200,
        "n_relevants": [10, 50, 80],
        "mixture_proportions": [0.5, 0.3, 0.2],
        "means": [0, 2, -2]
    }   
