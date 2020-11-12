#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf

from typing import Any, List
from gpflow import kernels
from gpflow.config import default_float
from gpflow.utilities import Dispatcher

# ---- Exports
__all__ = ('bias_initializer', 'weight_initializer')


# ==============================================
#                                   initializers
# ==============================================
MaternKernel = kernels.Matern12, kernels.Matern32, kernels.Matern52
bias_initializer = Dispatcher("bias_initializer")
weight_initializer = Dispatcher("weight_initializer")


@bias_initializer.register(kernels.Stationary, int)
def _bias_initializer_fallback(kern: kernels.Stationary,
                               ndims: int,
                               *,
                               batch_shape: List = None,
                               dtype: Any = None,
                               maxval: float = 2 * np.pi) -> tf.Tensor:
  if dtype is None:
    dtype = default_float()

  shape = [ndims] if batch_shape is None else list(batch_shape) + [ndims]
  return tf.random.uniform(shape=shape, maxval=maxval, dtype=dtype)


@weight_initializer.register(kernels.SquaredExponential, int)
def _weight_initializer_squaredExp(kern: kernels.SquaredExponential,
                                   ndims: int,
                                   *,
                                   batch_shape: List = None,
                                   dtype: Any = None,
                                   normal_rvs: tf.Tensor = None) -> tf.Tensor:
  if dtype is None:
    dtype = default_float()

  if batch_shape is None:
    batch_shape = []

  shape = [ndims] if batch_shape is None else list(batch_shape) + [ndims]
  if normal_rvs is None:
    return tf.random.normal(shape, dtype=dtype)

  assert tuple(normal_rvs.shape) == tuple(shape)
  return tf.convert_to_tensor(normal_rvs, dtype=dtype)


@weight_initializer.register(MaternKernel, int)
def _weight_initializer_matern(kern: MaternKernel,
                               ndims: int,
                               *,
                               batch_shape: List = None,
                               dtype: Any = None,
                               normal_rvs: tf.Tensor = None,
                               gamma_rvs: tf.Tensor = None) -> tf.Tensor:
  if dtype is None:
    dtype = default_float()

  if isinstance(kern, kernels.Matern12):
    smoothness = 1/2
  elif isinstance(kern, kernels.Matern32):
    smoothness = 3/2
  elif isinstance(kern, kernels.Matern52):
    smoothness = 5/2
  else:
    raise NotImplementedError

  batch_shape = [] if batch_shape is None else list(batch_shape)
  if normal_rvs is None:
    normal_rvs = tf.random.normal(shape=batch_shape + [ndims], dtype=dtype)
  else:
    assert tuple(normal_rvs.shape) == tuple(batch_shape + [ndims])
    normal_rvs = tf.convert_to_tensor(normal_rvs, dtype=dtype)

  if gamma_rvs is None:
    gamma_rvs = tf.random.gamma(shape=batch_shape + [1],
                                alpha=smoothness,
                                beta=smoothness,
                                dtype=dtype)
  else:
    assert tuple(gamma_rvs.shape) == tuple(batch_shape + [1])
    gamma_rvs = tf.convert_to_tensor(gamma_rvs, dtype=dtype)

  # Return draws from a multivariate-t distribution
  return tf.math.rsqrt(gamma_rvs) * normal_rvs
