#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import Any
from gpflow import kernels as gpflow_kernels
from gpflow.base import TensorType
from gpflow.inducing_variables import InducingVariables
from gpflow_sampling import kernels, inducing_variables
from gpflow_sampling.bases.core import AbstractBasis
from gpflow_sampling.utils import (move_axis,
                                   expand_to,
                                   batch_tensordot,
                                   inducing_to_tensor)
from gpflow_sampling.bases.fourier_initializers import (bias_initializer,
                                                        weight_initializer)


# ---- Exports
__all__ = (
  'Dense',
  'MultioutputDense',
  'Conv2d',
  'Conv2dTranspose',
  'DepthwiseConv2d',
)


# ==============================================
#                                  fourier_bases
# ==============================================
class AbstractFourierBasis(AbstractBasis):
  def __init__(self,
               kernel: gpflow_kernels.Kernel,
               num_bases: int,
               initialized: bool = False,
               name: Any = None):
    super().__init__(initialized=initialized, name=name)
    self.kernel = kernel
    self._num_bases = num_bases

  @property
  def num_bases(self):
    return self._num_bases


class Dense(AbstractFourierBasis):
  def __init__(self,
               kernel: gpflow_kernels.Stationary,
               num_bases: int,
               weights: tf.Tensor = None,
               biases: tf.Tensor = None,
               name: str = None):
    super().__init__(name=name,
                     kernel=kernel,
                     num_bases=num_bases)
    self._weights = weights
    self._biases = biases

  def __call__(self, x: TensorType, **kwargs) -> tf.Tensor:
    self._maybe_initialize(x, **kwargs)
    if isinstance(x, InducingVariables):  # TODO: Allow this behavior?
      x = inducing_to_tensor(x)

    proj = tf.tensordot(x, self.weights, axes=[-1, -1])  # [..., B]
    feat = tf.cos(proj + self.biases)
    return self.output_scale * feat

  def initialize(self, x: TensorType, dtype: Any = None):
    if isinstance(x, InducingVariables):
      x = inducing_to_tensor(x)

    if dtype is None:
      dtype = x.dtype

    self._biases = bias_initializer(self.kernel, self.num_bases, dtype=dtype)
    self._weights = weight_initializer(self.kernel, x.shape[-1],
                                       batch_shape=[self.num_bases],
                                       dtype=dtype)

  @property
  def weights(self):
    if self._weights is None:
      return None
    return tf.math.reciprocal(self.kernel.lengthscales) * self._weights

  @property
  def biases(self):
    return self._biases

  @property
  def output_scale(self):
    return tf.sqrt(2 * self.kernel.variance / self.num_bases)


class MultioutputDense(Dense):
  def __call__(self, x: TensorType, multioutput_axis: int = None, **kwargs):
    self._maybe_initialize(x, **kwargs)
    if isinstance(x, InducingVariables):  # TODO: Allow this behavior?
      x = inducing_to_tensor(x)

    # Compute (batch) tensor dot product
    batch_axes = None if (multioutput_axis is None) else [0, multioutput_axis]
    proj = move_axis(batch_tensordot(self.weights,
                                     x,
                                     axes=[-1, -1],
                                     batch_axes=batch_axes), 1, -1)

    ndims = proj.shape.ndims
    feat = tf.cos(proj + expand_to(self.biases, axis=1, ndims=ndims))
    return expand_to(self.output_scale, axis=1, ndims=ndims) * feat  # [L, N, B]

  def initialize(self, x: TensorType, dtype: Any = None):
    if isinstance(x, InducingVariables):
      x = inducing_to_tensor(x)

    if dtype is None:
      dtype = x.dtype

    biases = []
    weights = []
    for kernel in self.kernel.latent_kernels:
      biases.append(
          bias_initializer(kernel, self.num_bases, dtype=dtype))

      weights.append(
          weight_initializer(kernel, x.shape[-1],
                             batch_shape=[self.num_bases],
                             dtype=dtype))

    self._biases = tf.stack(biases, axis=0)  # [L, B]
    self._weights = tf.stack(weights, axis=0)  # [L, B, D]

  @property
  def weights(self):
    if self._weights is None:
      return None

    num_lengthscales = None
    for kernel in self.kernel.latent_kernels:
      if kernel.ard:
        ls = kernel.lengthscales
        assert ls.shape.ndims == 1
        if num_lengthscales is None:
          num_lengthscales = ls.shape[0]
        else:
          assert num_lengthscales == ls.shape[0]

    inv_lengthscales = []
    for kernel in self.kernel.latent_kernels:
      inv_ls = tf.math.reciprocal(kernel.lengthscales)
      if not kernel.ard and num_lengthscales is not None:
        inv_ls = tf.fill([num_lengthscales], inv_ls)
      inv_lengthscales.append(inv_ls)

    # [L, 1, D] or [L, 1, 1]
    inv_lengthscales = expand_to(arr=tf.stack(inv_lengthscales),
                                 axis=1,
                                 ndims=self._weights.shape.ndims)

    return inv_lengthscales * self._weights

  @property
  def output_scale(self):
    variances = tf.stack([k.variance for k in self.kernel.latent_kernels])
    return tf.sqrt(2 * variances / self.num_bases)  # [L]


class Conv2d(AbstractFourierBasis):
  def __init__(self,
               kernel: kernels.Conv2d,
               num_bases: int,
               filters: tf.Tensor = None,
               biases: tf.Tensor = None,
               name: str = None):

    super().__init__(name=name,
                     kernel=kernel,
                     num_bases=num_bases)
    self._filters = filters
    self._biases = biases

  def __call__(self, x: TensorType) -> tf.Tensor:
    self._maybe_initialize(x)
    if isinstance(x, InducingVariables) or len(x.shape) == 4:
      conv = self.convolve(x)
    elif len(x.shape) > 4:  # allow for higher order batches
      x_4d = tf.reshape(x, [-1] + list(x.shape[-3:]))
      conv = self.convolve(x_4d)
      conv = tf.reshape(conv, list(x.shape[:-3]) + list(conv.shape[1:]))
    else:
      raise NotImplementedError
    return self.output_scale * tf.cos(conv + self.biases)

  def convolve(self, x: TensorType) -> tf.Tensor:
    if isinstance(x, inducing_variables.InducingImages):
      return tf.nn.conv2d(input=x.as_images,
                          filters=self.filters,
                          strides=(1, 1, 1, 1),
                          padding="VALID")
    return self.kernel.convolve(input=x, filters=self.filters)

  def initialize(self, x, dtype: Any = None):
    if isinstance(x, inducing_variables.InducingImages):
      x = x.as_images

    if dtype is None:
      dtype = x.dtype

    self._biases = bias_initializer(self.kernel.kernel,
                                    self.num_bases,
                                    dtype=dtype)

    patch_size = (self.kernel.channels_in
                  * self.kernel.patch_shape[0]
                  * self.kernel.patch_shape[1])

    weights = weight_initializer(self.kernel.kernel, patch_size,
                                 batch_shape=[self.num_bases],
                                 dtype=dtype)

    shape = self.kernel.patch_shape + [self.kernel.channels_in, self.num_bases]
    self._filters = tf.reshape(move_axis(weights, -1, 0), shape)

  @property
  def filters(self):
    if self._filters is None:
      return None

    shape = list(self.kernel.patch_shape) + [self.kernel.channels_in, 1]
    inv_ls = tf.math.reciprocal(self.kernel.kernel.lengthscales)
    if self.kernel.kernel.ard:
      coeffs = tf.reshape(inv_ls, shape)
    else:
      coeffs = tf.fill(shape, inv_ls)

    return coeffs * self._filters

  @property
  def biases(self):
    return self._biases

  @property
  def output_scale(self):
    return tf.sqrt(2 * self.kernel.kernel.variance / self.num_bases)


class Conv2dTranspose(Conv2d):
  pass


class DepthwiseConv2d(Conv2d):
  def convolve(self, x: TensorType) -> tf.Tensor:
    if isinstance(x, inducing_variables.DepthwiseInducingImages):
      return tf.nn.depthwise_conv2d(input=x.as_images,
                                    filter=self.filters,
                                    strides=(1, 1, 1, 1),
                                    padding="VALID")

    return self.kernel.convolve(input=x, filters=self.filters)

  def initialize(self, x, dtype: Any = None):
    if isinstance(x, inducing_variables.InducingImages):
      x = x.as_images

    if dtype is None:
      dtype = x.dtype

    channels_out = self.kernel.channels_in * self.num_bases
    self._biases = bias_initializer(self.kernel.kernel,
                                    channels_out,
                                    dtype=dtype)

    patch_size = self.kernel.patch_shape[0] * self.kernel.patch_shape[1]
    batch_shape = [self.kernel.channels_in, self.num_bases]
    weights = weight_initializer(self.kernel.kernel, patch_size,
                                 batch_shape=batch_shape,
                                 dtype=dtype)

    self._filters = tf.reshape(move_axis(weights, -1, 0),
                               self.kernel.patch_shape + batch_shape)

  @property
  def filters(self):
    if self._filters is None:
      return None

    shape = list(self.kernel.patch_shape) + [self.kernel.channels_in, 1]
    inv_ls = tf.math.reciprocal(self.kernel.kernel.lengthscales)
    if self.kernel.kernel.ard:
      coeffs = tf.reshape(tf.transpose(inv_ls), shape)
    else:
      coeffs = tf.fill(shape, inv_ls)

    return coeffs * self._filters

  @property
  def output_scale(self):
    num_features_out = self.num_bases * self.kernel.channels_in
    return tf.sqrt(2 * self.kernel.kernel.variance / num_features_out)
