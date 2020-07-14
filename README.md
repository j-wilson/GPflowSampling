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

**Random Fourier features** are currently available for Matérn kernels, e.g. `<Matern52>` and `<SquaredExponential>`.
```
from gpflow_sampling.utils import RandomFourierBasis
kernel = gpflow.kernels.Matern52()
rff = RandomFourierBasis(kernel, num_basis=1024)
```


Naïve **Fourier-feature-based sampling** is provided for `<gpflow.model.GPR>`.
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

## Tips
As a running example, assume that `model` is multioutput GP accepting 3-dimensional inputs and predicting 2-dimensional function values. For  `<decoupled>` and `<finite_fourier>` samplers, the following tips apply:
- The third-to-last axis `axis=-3` acts as a batch axis for sample paths:
    ```
    sampler = decoupled(model, model.kernel, sample_shape=[32], num_basis=500
    from_2d = sampler(inputs=tf.random.uniform([100, 3]))  # 32x100x2, broadcasted evaluation
    from_3d = sampler(inputs=tf.random.uniform([32, 100, 3]))  # 32x100x2, pathwise evaluation
    ```
- We recommend use of custom mean functions, since GPflow's standard options do not broadcast --- which will cause pathwise evaluations like that of `from_3d` (above) to fail.
- `sampler.reset_random_variables(reset_basis=True)` can be used to resample paths without recreating `<tf.Variable>`. This is useful when `model` may have changed (see [examples/training_via_sampling.ipynb](examples/training_via_sampling.ipynb)).
- When generating samples in batches, typically use `sampler.reset_random_variables(reset_basis=False)`:
    ```
    a = []  # draws from a Gaussian distribution
    b = []  # draws from a Gaussian mixture
    x = tf.random.uniform([100, 3])
    for _ in range(10):
        sampler_a.reset_random_variables(reset_basis=False)
        a.append(sampler_a(x))  # drawn from the same Gaussian 

        sampler_b.reset_random_variables(reset_basis=True)
        b.append(sampler_b(x))  # drawn from a new Gaussian
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
