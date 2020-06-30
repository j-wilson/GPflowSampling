#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================

# ---- Imports
import numpy as np
import tensorflow as tf

from abc import abstractmethod
from typing import List, Union
from gpflow import kernels
from gpflow.base import TensorType
from gpflow.inducing_variables import InducingPoints
from tensorflow.python.keras.layers import Dense

# ---- Exports
__all__ = (
  'KernelBasis',
  'RandomFourierBasis',
)


# ==============================================
#                                basis_functions
# ==============================================
class AbstractBasis(tf.Module):
  @abstractmethod
  def __call__(self, *args, **kwargs):
    raise NotImplementedError

  def reset_random_variables(self):
    pass


class KernelBasis(AbstractBasis):
  def __init__(self,
               kernel: kernels.Kernel,
               centers: Union[TensorType, InducingPoints],
               name: str = None):

    super().__init__(name=name)
    self.kernel = kernel
    self.centers = centers

  def __call__(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
    if isinstance(self.centers, InducingPoints):       # [!] Improve me
      return self.kernel(x, self.centers.Z, **kwargs)  # GPflow does not
    return self.kernel(x, self.centers, **kwargs)      # broadcast this way


class RandomFourierBasis(AbstractBasis):
  def __init__(self,
               kernel: kernels.Stationary,
               units: int,
               layer: Dense = None,
               name: str = None,
               **layer_kwargs):
    """
    The term 'kernel' has two different uses here. Within the context
    of the layer, it refers to the weights. Otherwise, it denotes a
    kernel function such as a member of the Matern family.
    """
    super().__init__(name=name)
    self._kernel = kernel
    self._layer = layer
    self._units = units
    if layer is None:  # assign some default keyword arguments
      layer_kwargs.setdefault('units', units)
      layer_kwargs.setdefault('activation', tf.cos)
    else:
      assert len(layer_kwargs) == 0
    self._layer_kwargs = layer_kwargs  # passed when initializing layer

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    if self.layer is None:  # lazy init
      flag_layer_init = True
      self._layer = Dense(bias_initializer=self.bias_initializer,
                          kernel_initializer=self.kernel_initializer,
                          **self._layer_kwargs)
    else:
      flag_layer_init = False

    _inputs = tf.divide(inputs, self.kernel.lengthscales)
    outputs = self.layer(_inputs)

    if flag_layer_init:
      for var in self.layer.variables:
        var._trainable = False

    return tf.sqrt(2 * self.kernel.variance / self.units) * outputs

  def bias_initializer(self,
                       shape: List,
                       maxval: float = 2*np.pi,
                       **kwargs) -> tf.Tensor:
    return tf.random.uniform(shape=shape, maxval=maxval, **kwargs)

  def kernel_initializer(self, shape: List, **kwargs) -> tf.Tensor:
    _shape = list(shape)
    if isinstance(self.kernel, kernels.SquaredExponential):
      return tf.random.normal(_shape, **kwargs)
    elif isinstance(self.kernel, kernels.Matern52):
      normal_rvs = tf.random.normal(shape=_shape, **kwargs)
      gamma_rvs = tf.random.gamma(shape=_shape[:-2] + [1] + _shape[-1:],
                                  alpha=2.5, beta=2.5, **kwargs)
      return tf.math.rsqrt(gamma_rvs) * normal_rvs
    elif isinstance(self.kernel, kernels.Matern32):
      normal_rvs = tf.random.normal(shape=_shape, **kwargs)
      gamma_rvs = tf.random.gamma(shape=_shape[:-2] + [1] + _shape[-1:],
                                  alpha=1.5, beta=1.5, **kwargs)
      return tf.math.rsqrt(gamma_rvs) * normal_rvs
    elif isinstance(self.kernel, kernels.Matern12):
      normal_rvs = tf.random.normal(shape=_shape, **kwargs)
      gamma_rvs = tf.random.gamma(shape=_shape[:-2] + [1] + _shape[-1:],
                                  alpha=0.5, beta=0.5, **kwargs)
      return tf.math.rsqrt(gamma_rvs) * normal_rvs
    else:
      raise NotImplementedError

  def reset_random_variables(self):
    w, b = self.layer.variables
    self.layer.variables[0].assign(self.kernel_initializer(w.shape, dtype=w.dtype))
    self.layer.variables[1].assign(self.bias_initializer(b.shape, dtype=b.dtype))

  @property
  def layer(self):
    return self._layer

  @property
  def kernel(self):
    return self._kernel

  @property
  def units(self):
    return self._units
