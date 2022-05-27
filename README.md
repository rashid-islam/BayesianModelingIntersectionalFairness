# Bayesian Modeling of Intersectional Fairness
This repository provides code that implements empirical and model-based intersectional fairness estimation on COMPAS dataset from our paper: Bayesian Modeling of Intersectional Fairness: The Variance of Bias.

## Prerequisites

* Python
* PyMC3
* Other commonly used Python libraries: NumPy, pandas, and scikit-learn

The code is tested on windows and linux operating systems. It should work on any other platform.

## Instructions
Simply run the script "modeling_Intersectional_Fairness.py". It will generate necessary probabilities like p(y|S) and p(g(x)=1) which can be used to compute differential fairness and subgroup fairness (functions to compute these metrics are provided in the script). The probabilities p(y|S) and p(g(x)=1) will be generated for both point estimate and fully Bayesian version for all models like EDF, NB, LR, NN, and HLR.  

## Author

* Rashidul Islam (email: islam.rashidul@umbc.edu)

## Reference Paper

[Foulds, J.R., Islam, R., Keya, K.N. and Pan, S., 2020. Bayesian Modeling of Intersectional Fairness: The Variance of Bias. In Proceedings of the 2020 SIAM International Conference on Data Mining (pp. 424-432). Society for Industrial and Applied Mathematics.](https://epubs.siam.org/doi/epdf/10.1137/1.9781611976236.48).
