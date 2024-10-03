# VBVarSel

The goal of this package is to quickly and efficiently identify clusters of variables by using a scalable, computationally efficienty annealed variational Bayes algorithm for fitting high-dimensional mixture models with variable selection.

![equation](https://latex.codecogs.com/svg.image?\textit{p}(\textit{X}|\Phi,\pi)=\prod_{\textit{n=1}}^{\textit{N}}\prod_{\textit{k=1}}^{\textit{K}}\pi_k\mathit{f}\textsc{x}(\textsc{x}_n|\Phi_k))

For more information and to read the full research paper please visit [LINK]

## Installation
The VBVarSel package can be installed from github using pip:

`pip install git+https://github.com/<REPO>/PROJECT/#egg=<project name>`

or from PyPI itself:

`pip install vbvarsel`

## Using the package

### Parameters for simulation
Parameters can be left to optional default values or may be customised by the developer.

#### Simulation Parameters

Simulation parameters are parameters for the experiment itself, such as how many observations, the mixture proportion, number of variables to look for, etc. 

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

Hyperparameters affect equation itself, such as how many iterations the model will have, the annealing temperature, the threshold for the convergence and so on. More information on the hyperparameters can be found within the docstrings. These as well have default values, but can be altered by the user if desired. 

### Entry point

The packages entry point is `vbvarsel.main()`, and this where all the parameters will be passed. If they are not passed, they will be generated using default values. Users may supply their own data to use in the package. If no data is provided, data will be simulated according to [Crook et al (2019)](https://pubmed.ncbi.nlm.nih.gov/31119032/)'s methodology. 

Data is processed through the simulation to identify clustering of relevant data. An optional `save_output` parameter can be passed to save the data, as well as a path to the targeted directory. If no path is provided, data will be saved in the current working directory.

```
from vbvarsel import vbvarsel

sim_params = vbvarsel.SimulationParameters()
hyp_params = vbvarsel.Hyperparameters()

vbarsel.main(sim_params, hyp_params)
```

### Contributing

If you are interested in contributing to this package, please submit a pull request.

#### Future implementations

A CLI command suite.

### Issues

If you come across an issue when using this package, please create an issue on the issues page and someone will respond to it as soon as we can.

### License

This project is developed by the MRC-Biostatistical Unit at Cambridge University under the GNU Public license.