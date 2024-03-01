# Sparse Component Analysis: An unsupervised method for recovering interpretable latent factors
![SCA_overview_v1.pdf](https://github.com/andrewZimnik/sca/files/14455291/SCA_overview_v1.pdf)

## Overview
Sparse Component Analysis (SCA) is an unsupervised dimensionality reduction method that can recover interpretable latent factors from high dimensional neural activity. This repo. contains notebooks (and example data) that demonstrate how to use SCA. For a full characterization of the method (and to see results from additional datasets) see the [preprint](https://www.biorxiv.org/content/10.1101/2024.02.05.578988v1).

## Installation and Dependencies

This package can be installed by: 
```buildoutcfg
git clone https://github.com/glaserlab/sca.git
cd sca
pip install -e .
```
This package requires python 3.6 or higher for a succesful installation.

## Getting started

Let's say we have a matrix **X** that contains the activity of *N* neurons over *T* time points (it is dimension *T* x *N*). We want to reduce the dimensionality to *K* instead of *N*.

To do this, we first import the necessary function and then run SCA:
```python
from sca.models import SCA
sca = SCA(n_components=K)
latent = sca.fit_transform(X)
```

Please see the example jupyter notebook **`Example_1pop.ipynb`** in the notebooks folder for further details on a simulated dataset. <br>
Also see the example notebooks **`centerOutReaching.ipnyb`**, and **`unimanualCycling.ipynb`**  for examples using neural data from our manuscript. <br><br>

## SCA in a nutshell
Many neural computations are composed of multiple, distinct processes, each associated with different sets of latent factors. 

For example, in a center-out reaching task, reach-generating computations in motor cortex are composed of [preparatory](https://elifesciences.org/articles/31826) and [execution-related](https://www.nature.com/articles/ncomms13239) processes. Preparation and execution can occur at different times, in nearly-orthogonal sets of dimensions. Preparatory and execution-related factors can therefore be recovered if we look for factors that are temporally sparse and active in orthogonal sets of dimensions. 

This is how SCA works; it looks for temporally sparse factors that evolve in orthogonal dimensions (which can also reconstruct single-unit activity). Note that these assumptions are not specific to reaching. We found that the same cost function produces interpretable factors across a wide variety of tasks involving a variety of experimental models (monkeys, *C. elegans*, and artificial networks). In all cases, SCA recovered factors that reflected a single computational role. Importantly, this interpretability did not require any supervision; SCA works without needing to know anything about the task structure. 


## Usage notes
### Hyperparameters
SCA has three hyperparameters. Results are generally robust to values of these hyperparameters (see preprint, including Extended Data Fig. 4). Here's what you can expect from changing each of the hyperparameters, and our general recommendations.

  1. lam_sparse: determines how much to penalize non-sparse factors. While our default value generally works well, we have found that setting lam_sparse as high as possible, without sacrificing reconstruction error, generally led to slightly improved qualitative results (relative to the default). See our example notebook **`Example_sparsity_hyperparm.ipynb`**, for a demonstration of how to do this selection. 

  2. lam_orthog: determines how much non-orthogonal dimensions are penalized (higher values leads to greater orthogonality). We recommend using the default value, which was used for all datasets in our preprint. Additionally, we do have an option for using a strict orthogonality constraint rather than the penalty, but n general, allowing SCA dimensions to be orthogonal-ish (rather than requiring them to be strictly orthogonal) led to slightly cleaner (and faster) results.   

  3. n_components: number of requested factors. SCA will not 'over-sparsify' data when increasing the number of requested factors. In other words, SCA will not produce two factors that have very similar loading vectors (this behavior is documented in the preprint, Extended Data Fig. 4). The orthogonality term in the cost function prevents this. That said, the neural activity relating to two different processes is not always completely aligned or completely unaligned. For example, the dimensions occupied by motor cortical activity during forwards cycling are sometimes partially aligned with those occupied during backwards cycling (see the preprint, Extended Data Fig. 14). In these cases, requesting 'too few' dimensions may cause SCA to find a factor that is active during both forwards and backwards cycling, while requesting more dimensions could yield two different factors, one related to each behavior. In general, it is useful to run SCA multiple times, requesting increasing numbers of factors. Doing so will give you a sense for which component processes use completely separate, partially overlapping, or largely overlapping dimensions. 
