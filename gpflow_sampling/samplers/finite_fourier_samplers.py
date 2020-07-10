#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
import gpflow

from typing import List, Callable, Any
from gpflow.config import default_float
from tensorflow_probability.python.distributions import MultivariateNormalDiag
from gpflow_sampling.utils import RandomFourierBasis, BayesianLinearRegression
from gpflow_sampling.samplers.base import BayesianLinearSampler
from gpflow_sampling.samplers.dispatch import finite_fourier

# ---- Exports
__all__ = tuple()


# ==============================================
#                        finite_fourier_samplers
# ==============================================
@finite_fourier.register(gpflow.models.GPR, gpflow.kernels.Stationary)
def _finite_fourier_gpr(model: gpflow.models.GPR,
                        kernel: gpflow.kernels.Stationary,
                        sample_shape: List[int],
                        num_basis: int,
                        basis: Callable = None,
                        prior: MultivariateNormalDiag = None,
                        dtype: Any = None,
                        **kwargs):
  if dtype is None:
    dtype = default_float()

  if basis is None:
    basis = RandomFourierBasis(kernel=model.kernel,
                               units=num_basis,
                               dtype=dtype)

  if prior is None:
    prior = MultivariateNormalDiag(scale_diag=tf.ones(num_basis, dtype=dtype))

  blr = BayesianLinearRegression(basis=basis,
                                 prior=prior,
                                 likelihood=model.likelihood)

  X, y = model.data
  if model.mean_function is not None:
    y = y - model.mean_function(X)

  def initializer(shape, dtype):
    weights = blr.predict_w_samples(sample_shape=shape[:-1], data=(X, y))
    assert weights.shape[-1] == shape[-1] == basis.units
    return tf.cast(weights, dtype)

  weight_shape = list(sample_shape) + [1, num_basis]
  weights = initializer(weight_shape, dtype)
  return BayesianLinearSampler(basis=basis,
                               weights=weights,
                               mean_function=model.mean_function,
                               weight_initializer=initializer,
                               **kwargs)
