#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
import gpflow

from string import ascii_lowercase
from typing import List
from gpflow.kernels import SeparateIndependent, \
                           SharedIndependent, \
                           LinearCoregionalization
from gpflow_sampling.samplers.base import CompositeSampler
from gpflow_sampling.samplers.dispatch import decoupled
from gpflow_sampling.samplers.decoupled_samplers import _decoupled_sampler_svgp

IndependentMultioutputKernel = SharedIndependent, SeparateIndependent

# ---- Exports
__all__ = tuple()


# ==============================================
#                             decoupled_samplers
# ==============================================
@decoupled.register(gpflow.models.SVGP, IndependentMultioutputKernel)
def _decoupled_sampler_svgp_mo_independent(model: gpflow.models.SVGP,
                                           kernel: IndependentMultioutputKernel,
                                           sample_shape: List[int],
                                           num_basis: int,
                                           **kwargs):

  def join_rule(samples: List[tf.Tensor]) -> tf.Tensor:
    return tf.concat(samples, axis=-1)

  samplers = []
  for latent_dim in range(model.kernel.num_latent_gps):
    samplers.append(_decoupled_sampler_svgp(model=model,
                                            kernel=None,  # taken from the model
                                            mean_function='None',  # apply after
                                            sample_shape=sample_shape,
                                            num_basis=num_basis,
                                            latent_dim=latent_dim,
                                            **kwargs))

  return CompositeSampler(join_rule=join_rule,
                          samplers=samplers,
                          mean_function=model.mean_function)


@decoupled.register(gpflow.models.SVGP, LinearCoregionalization)
def _decoupled_sample_svgp_lcm(model: gpflow.models.SVGP,
                               kernel: LinearCoregionalization,
                               sample_shape: List[int],
                               num_basis: int,
                               **kwargs):

  def join_rule(samples: List[tf.Tensor]) -> tf.Tensor:
    """
    Computes the matrix multiply: [a, b] x [b, ..., 1] -> [..., a]
    """
    ndims = samples[0].shape.ndims - 1
    lhs_w = ascii_lowercase[:2]
    lhs_s = lhs_w[-1:] + ascii_lowercase[2: 2 + ndims]
    rhs = lhs_s[1:] + lhs_w[:1]
    return tf.einsum(f'{lhs_w},{lhs_s}->{rhs}',
                     model.kernel.W, tf.squeeze(samples, axis=-1))

  samplers = []
  for latent_dim in range(model.kernel.num_latent_gps):
    samplers.append(_decoupled_sampler_svgp(model=model,
                                            kernel=None,  # taken from the model
                                            mean_function='None',  # apply after
                                            sample_shape=sample_shape,
                                            num_basis=num_basis,
                                            latent_dim=latent_dim,
                                            **kwargs))

  return CompositeSampler(join_rule=join_rule,
                          samplers=samplers,
                          mean_function=model.mean_function)