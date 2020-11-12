#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import List, Callable
from gpflow import inducing_variables
from gpflow.base import TensorLike
from gpflow.utilities import Dispatcher
from gpflow.kernels import Kernel, MultioutputKernel, LinearCoregionalization
from gpflow_sampling.sampling.updates import exact as exact_update
from gpflow_sampling.sampling.core import AbstractSampler, CompositeSampler
from gpflow_sampling.kernels import Conv2d
from gpflow_sampling.inducing_variables import InducingImages

# ---- Exports
__all__ = ('decoupled',)
decoupled = Dispatcher("decoupled")


# ==============================================
#                             decoupled_samplers
# ==============================================
@decoupled.register(Kernel, AbstractSampler, TensorLike, TensorLike)
def _decoupled_fallback(kern: Kernel,
                        prior: AbstractSampler,
                        Z: TensorLike,
                        u: TensorLike,
                        *,
                        mean_function: Callable = None,
                        update_rule: Callable = exact_update,
                        join_rule: Callable = sum,
                        **kwargs):

  f = prior(Z, sample_axis=None)  # [S, M, L]
  update = update_rule(kern, Z, u, f, **kwargs)
  return CompositeSampler(samplers=[prior, update],
                          join_rule=join_rule,
                          mean_function=mean_function)


@decoupled.register(MultioutputKernel, AbstractSampler, TensorLike, TensorLike)
def _decoupled_multioutput(kern: MultioutputKernel,
                           prior: AbstractSampler,
                           Z: TensorLike,
                           u: TensorLike,
                           *,
                           mean_function: Callable = None,
                           update_rule: Callable = exact_update,
                           join_rule: Callable = sum,
                           multioutput_axis_Z: int = "default",
                           **kwargs):

  # Determine whether or not to evalaute Z pathwise (per output feature)
  # TODO: Ugly. This argument is actually being passed to the prior's basis.
  #       Disallow non-inducing-variable Z for multioutput cases of decoupled?
  if multioutput_axis_Z == "default":
    if isinstance(Z, inducing_variables.MultioutputInducingVariables) and not\
       isinstance(Z, inducing_variables.SharedIndependentInducingVariables):
      multioutput_axis_Z = 0
    else:
      multioutput_axis_Z = None

  f = prior(Z, sample_axis=None, multioutput_axis=multioutput_axis_Z)
  update = update_rule(kern, Z, u, f, **kwargs)
  return CompositeSampler(samplers=[prior, update],
                          join_rule=join_rule,
                          mean_function=mean_function)


@decoupled.register(LinearCoregionalization, AbstractSampler, TensorLike, TensorLike)
def _decoupled_lcm(kern: LinearCoregionalization,
                   prior: AbstractSampler,
                   Z: TensorLike,
                   u: TensorLike,
                   *,
                   join_rule: Callable = None,
                   **kwargs):
  if join_rule is None:
    def join_rule(terms: List[tf.Tensor]) -> tf.Tensor:
      return tf.tensordot(kern.W, sum(terms), axes=[-1, 0])
  return _decoupled_multioutput(kern, prior, Z, u, join_rule=join_rule, **kwargs)


@decoupled.register(Conv2d, AbstractSampler, InducingImages, TensorLike)
def _decoupled_conv(kern: Conv2d,
                    prior: AbstractSampler,
                    Z: InducingImages,
                    u: TensorLike,
                    *,
                    mean_function: Callable = None,
                    update_rule: Callable = exact_update,
                    join_rule: Callable = sum,
                    **kwargs):

  f = tf.squeeze(prior(Z, sample_axis=None), axis=[-3, -2])  # [S, M, L]
  update = update_rule(kern, Z, u, f, **kwargs)
  return CompositeSampler(samplers=[prior, update],
                          join_rule=join_rule,
                          mean_function=mean_function)
