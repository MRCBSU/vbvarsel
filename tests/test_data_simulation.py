# import sys
# sys.path.append(r"C:\Users\Alan\Desktop\dev\variationalTempering_beta\src")
from vbvarsel.global_parameters import Hyperparameters, SimulationParameters

def test_default_hyperparameters():
    default_hypers = Hyperparameters()
    assert default_hypers.max_models == 10

def test_establish_sim_params():
    default_sim_params = SimulationParameters()
    assert default_sim_params.mixture_proportions == [0.5, 0.3, 0.2]
