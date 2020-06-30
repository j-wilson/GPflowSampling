# GPflowSampling
Companion code for [Efficiently Sampling Functions from Gaussian Process Posteriors](https://arxiv.org/abs/2002.09309). This software has been provided on an "as is" basis and depends heavily upon [GPflow](https://github.com/GPflow/GPflow). 


## Installation
```
git clone https://github.com/j-wilson/GPflowSampling
cd GPflowSampling
pip install -e .
```
To install the dependencies needed to run `examples`, use `pip install -e .[examples]`.


## Usage
**Location-scale-based sampling** is implemented for scalar output `<gpflow.model.GPR>` and `<gpflow.model.SVGP>`.
```
from gpflow_sampling.samplers import location_scale
model = gpflow.model.GPR(...)
sampler = location_scale(model, model.kernel, sample_shape=[...], full_cov=True)
```

**Random Fourier features** are currently available Matérn kernels, e.g. `<Matern52>` and `<SquaredExponential>`.
```
from gpflow_sampling.utils import RandomFourierBasis
kernel = gpflow.kernels.Matern52()
rff = RandomFourierBasis(kernel, num_basis=1024)
```


Naïve **Fourier-feature-based sampling** is provided for `<<gpflow.model.GPR>`.
```
from gpflow_sampling.samplers import finite_fourier
model = gpflow.model.GPR(...)
sampler = finite_fourier(model, model.kernel, sample_shape=[...], num_basis=5000)
```


**Decouple sampling** is supported for `<gpflow.model.GPR>` and `<gpflow.model.SVGP>`. Additional coverage is provided for some simple multi-output kernels.
```
from gpflow_sampling.samplers import decoupled
kernel = gpflow.kernels.LinearCoregionalization(...)
model = gpflow.model.SVGP(kernel=kernel, ...)
sampler = decoupled(model, model.kernel, sample_shape=[...], num_basis=500)
```



## Citing Us
If our work helps you in a way that you feel warrants reference, please cite the following paper:
```
@inproceedings{wilson2020efficiently,
    title={Efficiently sampling functions from Gaussian process posteriors},
    author={James T. Wilson
            and Viacheslav Borovitskiy
            and Alexander Terenin
            and Peter Mostowsky
            and Marc Peter Deisenroth},
    booktitle={International Conference on Machine Learning},
    year={2020},
    url={https://arxiv.org/abs/2002.09309}
}
```