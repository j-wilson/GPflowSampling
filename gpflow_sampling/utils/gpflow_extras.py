#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel, \
                           MultioutputKernel, \
                           SeparateIndependent, \
                           SharedIndependent, \
                           Matern52 as GPflowMatern52

from gpflow.inducing_variables import InducingVariables, \
                                       MultioutputInducingVariables, \
                                       SharedIndependentInducingVariables, \
                                       SeparateIndependentInducingVariables

from gpflow.utilities import Dispatcher
slice_multioutput_kernel = Dispatcher("slice_multioutput_kernel")
slice_multioutput_inducing = Dispatcher("slice_multioutput_inducing")

# ---- Exports
__all__ = (
  'Matern52',
  'slice_multioutput_kernel',
  'slice_multioutput_inducing',
)


# ==============================================
#                                  gpflow_extras
# ==============================================
class Matern52(GPflowMatern52):
  def K_r2(self, r2, eps=None):  # faster than <gpflow.kernels.Matern52.K_r2>
    if eps is None:
      eps = np.finfo(r2.dtype.as_numpy_dtype).eps

    _r2 = 5.0 * tf.clip_by_value(r2, eps, float('inf'))
    _r = tf.sqrt(_r2)
    return self.variance * (1.0 + _r + (1 / 3) * _r2) * tf.exp(-_r)


@slice_multioutput_kernel.register(Kernel, int)
def _getter(kernel, latent_dim):
  assert not isinstance(MultioutputKernel)
  assert latent_dim == 0
  return kernel


@slice_multioutput_kernel.register(SharedIndependent, int)
def _getter(kernel, latent_dim):
  if latent_dim < 0:
    latent_dim = kernel.num_latent_gps + latent_dim

  assert kernel.num_latent_gps > latent_dim >= 0
  return kernel.kernel


@slice_multioutput_kernel.register(SeparateIndependent, int)
def _getter(kernel, latent_dim):
  if latent_dim < 0:
    latent_dim = kernel.num_latent_gps + latent_dim

  assert kernel.num_latent_gps > latent_dim >= 0
  return kernel.latent_kernels[latent_dim]


@slice_multioutput_inducing.register(InducingVariables, int)
def _getter(inducing_variable, latent_dim):
  assert not isinstance(inducing_variable, MultioutputInducingVariables)
  return inducing_variable


@slice_multioutput_inducing.register(SharedIndependentInducingVariables, int)
def _getter(inducing_variable, latent_dim):
  return inducing_variable.inducing_variable


@slice_multioutput_inducing.register(SeparateIndependentInducingVariables, int)
def _getter(inducing_variable, latent_dim):
  return inducing_variable.inducing_variable_list[latent_dim]
