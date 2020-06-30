#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import *
from gpflow.config import default_float
from gpflow.kernels import Stationary
from gpflow_sampling.utils.basis_functions import RandomFourierBasis

# ---- Exports
__all__ = ('RandomFourierTask',)


# ==============================================
#                                          tasks
# ==============================================
class RandomFourierTask:
  def __init__(self,
               kernel: Stationary,
               input_dim: int,
               variance: float = None,
               num_basis: int = 16384,
               dtype: Any = None):

    if dtype is None:
      dtype = default_float()

    self.input_dim = input_dim
    self.basis = RandomFourierBasis(kernel=kernel,
                                    units=num_basis,
                                    dtype=dtype)

    self.weights = tf.random.normal([num_basis, 1], dtype=dtype)
    self.variance = variance

  def __call__(self, x, noisy=True):
    f = tf.matmul(self.basis(x), self.weights)
    if noisy and self.variance is not None:
      f += (self.variance ** 0.5) * tf.random.normal(f.shape, dtype=f.dtype)
    return f
