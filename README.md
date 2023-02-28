# SCA: Sparse Component Analysis

This is an in-development package for "sparse component analysis," a dimensionality reduction tool that aims to provide more interpretable low-D representations than PCA. Note that we previously referred to the method as "ssa", and have not yet changed the acronym within the package.

## Installation and Dependencies

This package can be installed by: 
```buildoutcfg
git clone https://github.com/jglaser2/sca.git
cd ssa
pip install -e .
```
This package requires python 3.6 or higher for a succesful installation.


## Getting started

Let's say we have a matrix **X** that contains the activity of *N* neurons over *T* time points (it is dimension *T* x *N*). We want to reduce the dimensionality to *R* instead of *N*.

To do this, we first import the necessary function and then run SSA:
```python
from ssa.models import fit_ssa
model, latent, x_hat, losses = fit_ssa(X,R)
```
where the output "latent" is the low-D representation and "x_hat" is the reconstructed high-D data.


Please see the example jupyter notebook **`Example_1pop.ipynb`** in the notebooks folder for further details. <br><br>


Additionally, the example notebook **`Example_2pops.ipynb`** demonstrates how to find interpretable low-D representations between two populations. <br>
