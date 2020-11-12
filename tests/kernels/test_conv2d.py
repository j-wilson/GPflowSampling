#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from numpy import allclose
from typing import Any, List, NamedTuple
from gpflow import config as gpflow_config, kernels as gpflow_kernels
from gpflow.config import default_float as floatx
from gpflow_sampling import kernels, covariances
from gpflow_sampling.inducing_variables import *
from gpflow_sampling.covariances.Kfus import (_Kfu_conv2d_fallback,
                                              _Kfu_depthwise_conv2d_fallback)

SupportedBaseKernels = (gpflow_kernels.Matern12,
                        gpflow_kernels.Matern32,
                        gpflow_kernels.Matern52,
                        gpflow_kernels.SquaredExponential,)


# ==============================================
#                                    test_conv2d
# ==============================================
class ConfigConv2d(NamedTuple):
  seed: int = 1
  floatx: Any = 'float64'
  jitter: float = 1e-16

  num_test: int = 16
  num_cond: int = 32
  kernel_variance: float = 1.0
  rel_lengthscales_min: float = 0.1
  rel_lengthscales_max: float = 1.0


  channels_in: int = 3
  channels_out: int = 2
  image_shape: List = [28, 28]
  patch_shape: List = [5, 5]
  strides: List = [2, 2]

  padding: str = "SAME"
  dilations: List = [1, 1]


def test_conv2d(config: ConfigConv2d = None):
  if config is None:
    config = ConfigConv2d()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X_shape = [config.num_test] + config.image_shape + [config.channels_in]
  X = tf.reshape(tf.range(tf.reduce_prod(X_shape), dtype=floatx()), X_shape)
  X /= tf.reduce_max(X)

  patch_len = config.channels_in * int(tf.reduce_prod(config.patch_shape))
  for cls in SupportedBaseKernels:
    minval = config.rel_lengthscales_min * (patch_len ** 0.5)
    maxval = config.rel_lengthscales_max * (patch_len ** 0.5)
    lenscales = tf.random.uniform(shape=[patch_len],
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=floatx())

    base = cls(lengthscales=lenscales, variance=config.kernel_variance)
    kern = kernels.Conv2d(kernel=base,
                          image_shape=config.image_shape,
                          patch_shape=config.patch_shape,
                          channels_in=config.channels_in,
                          channels_out=config.channels_out,
                          dilations=config.dilations,
                          padding=config.padding,
                          strides=config.strides)

    kern._weights = tf.random.normal(kern._weights.shape, dtype=floatx())

    # Test full and shared inducing images
    Z_shape = [config.num_cond] + config.patch_shape + [config.channels_in]
    Zsrc = tf.random.normal(Z_shape, dtype=floatx())
    for Z in (InducingImages(Zsrc),
              SharedInducingImages(Zsrc[..., :1], config.channels_in)):

      test = _Kfu_conv2d_fallback(Z, kern, X)
      allclose(covariances.Kfu(Z, kern, X), test)


def test_conv2d_transpose(config: ConfigConv2d = None):
  if config is None:
    config = ConfigConv2d()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X_shape = [config.num_test] + config.image_shape + [config.channels_in]
  X = tf.reshape(tf.range(tf.reduce_prod(X_shape), dtype=floatx()), X_shape)
  X /= tf.reduce_max(X)

  patch_len = config.channels_in * int(tf.reduce_prod(config.patch_shape))
  for cls in SupportedBaseKernels:
    minval = config.rel_lengthscales_min * (patch_len ** 0.5)
    maxval = config.rel_lengthscales_max * (patch_len ** 0.5)
    lenscales = tf.random.uniform(shape=[patch_len],
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=floatx())

    base = cls(lengthscales=lenscales, variance=config.kernel_variance)
    kern = kernels.Conv2dTranspose(kernel=base,
                                   image_shape=config.image_shape,
                                   patch_shape=config.patch_shape,
                                   channels_in=config.channels_in,
                                   channels_out=config.channels_out,
                                   dilations=config.dilations,
                                   padding=config.padding,
                                   strides=config.strides)

    kern._weights = tf.random.normal(kern._weights.shape, dtype=floatx())

    # Test full and shared inducing images
    Z_shape = [config.num_cond] + config.patch_shape + [config.channels_in]
    Zsrc = tf.random.normal(Z_shape, dtype=floatx())
    for Z in (InducingImages(Zsrc),
              SharedInducingImages(Zsrc[..., :1], config.channels_in)):

      test = _Kfu_conv2d_fallback(Z, kern, X)
      allclose(covariances.Kfu(Z, kern, X), test)


def test_depthwise_conv2d(config: ConfigConv2d = None):
  if config is None:
    config = ConfigConv2d()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X_shape = [config.num_test] + config.image_shape + [config.channels_in]
  X = tf.reshape(tf.range(tf.reduce_prod(X_shape), dtype=floatx()), X_shape)
  X /= tf.reduce_max(X)

  patch_len = int(tf.reduce_prod(config.patch_shape))
  for cls in SupportedBaseKernels:
    minval = config.rel_lengthscales_min * (patch_len ** 0.5)
    maxval = config.rel_lengthscales_max * (patch_len ** 0.5)
    lenscales = tf.random.uniform(shape=[config.channels_in, patch_len],
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=floatx())

    base = cls(lengthscales=lenscales, variance=config.kernel_variance)
    kern = kernels.DepthwiseConv2d(kernel=base,
                                   image_shape=config.image_shape,
                                   patch_shape=config.patch_shape,
                                   channels_in=config.channels_in,
                                   channels_out=config.channels_out,
                                   dilations=config.dilations,
                                   padding=config.padding,
                                   strides=config.strides)

    kern._weights = tf.random.normal(kern._weights.shape, dtype=floatx())

    # Test full and shared inducing images
    Z_shape = [config.num_cond] + config.patch_shape + [config.channels_in]
    Zsrc = tf.random.normal(Z_shape, dtype=floatx())
    for Z in (DepthwiseInducingImages(Zsrc),
              SharedDepthwiseInducingImages(Zsrc[..., :1], config.channels_in)):

      test = _Kfu_depthwise_conv2d_fallback(Z, kern, X)
      allclose(covariances.Kfu(Z, kern, X), test)

