#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from gpflow.inducing_variables import (InducingVariables,
                                       MultioutputInducingVariables,
                                       SharedIndependentInducingVariables,
                                       SeparateIndependentInducingVariables)

from gpflow.utilities import Dispatcher


# ---- Exports
__all__ = ('get_inducing_shape', 'inducing_to_tensor')


# ==============================================
#                                   gpflow_utils
# ==============================================
get_inducing_shape = Dispatcher("get_inducing_shape")
inducing_to_tensor = Dispatcher("inducing_to_tensor")

@get_inducing_shape.register(InducingVariables)
def _getter(x):
  assert not isinstance(InducingVariables, MultioutputInducingVariables)
  return list(x.Z.shape)


@get_inducing_shape.register(SharedIndependentInducingVariables)
def _getter(x: SharedIndependentInducingVariables):
  assert len(x.inducing_variables) == 1
  return get_inducing_shape(x.inducing_variables[0])


@get_inducing_shape.register(SeparateIndependentInducingVariables)
def _getter(x: SeparateIndependentInducingVariables):
  for n, z in enumerate(x.inducing_variables):
    if n == 0:
      shape = get_inducing_shape(z)
    else:
      assert shape == get_inducing_shape(z)
  return [n + 1] + shape


@inducing_to_tensor.register(InducingVariables)
def _convert(x: InducingVariables, **kwargs):
  assert not isinstance(InducingVariables, MultioutputInducingVariables)
  return tf.convert_to_tensor(x.Z, **kwargs)


@inducing_to_tensor.register(SharedIndependentInducingVariables)
def _convert(x: InducingVariables, **kwargs):
  assert len(x.inducing_variables) == 1
  return inducing_to_tensor(x.inducing_variables[0], **kwargs)


@inducing_to_tensor.register(SeparateIndependentInducingVariables)
def _convert(x: InducingVariables, **kwargs):
  return tf.stack([inducing_to_tensor(z, **kwargs)
                   for z in x.inducing_variables], axis=0)

