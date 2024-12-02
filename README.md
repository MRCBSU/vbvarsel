# VBVarSel

The goal of this package is to quickly and efficiently identify clusters of variables by using a scalable, computationally efficienty annealed variational Bayes algorithm for fitting high-dimensional mixture models with variable selection.

The preprint for the underlying research paper can be found [here](https://arxiv.org/abs/2411.19262).

## Installation
The VBVarSel package can be installed from github using pip:

`pip install git+https://github.com/MRCBSU/PROJECT/#egg=vbvarsel`

or directly from PyPI:

`pip install vbvarsel`

## Using the package

### Parameters for simulation
Parameters can be left to optional default values or may be customised by the developer.

#### Simulation Parameters

Simulation parameters are parameters for simulating an experiment with synthetic data. Data is created synthetically according to Crook et al, [read the paper here](https://www.degruyter.com/document/doi/10.1515/sagmb-2018-0065/html). 

```
# default values for the simulation parameters.

n_observations: list[int] = [100,1000]
n_variables: int = 200
n_relevants: list[int] = [10, 20, 50, 100]
mixture_proportions: list[float] = [0.2, 0.3, 0.5]
means: list[int] = [-2, 0, 2]
```

Some things to note when customising parameters:

- No number in `n_relevants` should exceed the `n_variables` parameter. 
- `mixture_proportions` total values must sum to 1.0 exactly.

#### Hyperparameters

Hyperparameters affect equation itself, such as how many iterations the model will have, the annealing temperature, the threshold for the convergence and so on. More information on the hyperparameters can be found within the docstrings. These as well have default values, but can be altered by the user if desired. The default Hyperparameters are described below

```
#Threshold for the ELBO convergence
threshold = 1e-1

#Maximum number of mixture components
k1 = 5 

#Prior coefficient count for Dirichlet prior
alpha0 = 1/(K1) #cabassi

#Shrinkage parameter of the Gaussian conditional prior
beta0 = (1e-3)*1.

#Degrees of freedom for the Gamma prior
a0 = 3.
    
#Shape parameter of the Beta distribution
d = 1

#Maximum starting annealing temperature. The default value of 1 applies no annealing.
t_max = 1.
#NOTE: t_max CANNOT equal zero. There are several functions that divide or multiply by t_max. One cannot divide by zero.
#If you need to get very close to zero, just use a very small decimal.

#Maximum number of iterations for the simulation
max_itr = 25

#Maximum number of models
max_models = 10
```

### User-supplied data

Users may supply their own data, pending a few caveats. Data must be passed in by using a path to a file location, which is then loaded into a pandas DataFrame. Data used in the algorithm can only have numerical data. A set of labels (so-called "true labels") is preferred to verify accuracy via ARI (adjusted Rand index), but not required. If a dataset contains non-numerical data, these columns must be passed as the `cols_to_skip` parameter in `vbvarsel.main()`, and they will be dropped from the DataFrame before the algorithm commences. Users using their own data will not use any of the `SimulationParameters`, even if they are initialised they will be ignored. 


### Entry point

The packages entry point is `vbvarsel.main()`, and this where all the aforementioned experiment parameters will be passed. If they are not passed, they will be generated using default values or ignored in the case of user-supplied data. 

Data is processed through the simulation to identify clustering of relevant data. An optional `save_output` parameter can be passed to save the data to the current working directory. The simulation also returns a results object, if a user wishes to
use the output data for further uses. 

```
import vbvarsel as vbvs

sim_params = vbvs.SimulationParameters()
hyp_params = vbvs.Hyperparameters()

results = vbs.main(simulation_parameters=sim_params, hyperparameters=hyp_params)
#an optional `save_output` flag will save the output to the current working directory.
```

### Contributing

If you are interested in contributing to this package, please submit a pull request.

#### Future implementations

A CLI interface.

### Issues

If you come across an issue when using this package, please create an issue on the issues page and someone will respond to it as soon as we can.

### License

This project is developed by the MRC-Biostatistics Unit at Cambridge University under the GNU Public license.
