# PyTorch Bayesian Model Utilities
Set of Pytorch functions designed to be modular and transferable to any Neural Network application. Includes functionality for Bayesian models

## Environment Requirements
- PyTorch
- bayesian-torch
- torchmetics
- tqdm
- numpy
- pandas
- matplotlib

## Setup

1. Clone the git repository \
git clone https://github.com/brianelie/pytorch_model_utils.git

2. Set Up Environment \
Optionally, use the provided environment file (run from repo directory): \
conda env create -n ENVNAME --file environment.yml
Or at least ensure that packages liked in [Environmental Requirements](#environment-requirements) are in your environment. 

3. Import the functions into your codebase: \
from pytorch_model_utils.utils import train, learning_curves, evaluate, uncertainty_quantification
