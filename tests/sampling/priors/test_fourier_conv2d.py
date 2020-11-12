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
from gpflow_sampling import kernels, covariances, inducing_variables
from gpflow_sampling.sampling.priors import random_fourier
from gpflow_sampling.bases import fourier as fourier_basis
from gpflow_sampling.utils import batch_tensordot, swap_axes

SupportedBaseKernels = (gpflow_kernels.Matern12,
                        gpflow_kernels.Matern32,
                        gpflow_kernels.Matern52,
                        gpflow_kernels.SquaredExponential,)


# ==============================================
#                            test_fourier_conv2d
# ==============================================
class ConfigFourierConv2d(NamedTuple):
  seed: int = 1
  floatx: Any = 'float64'
  jitter: float = 1e-6

  num_test: int = 16
  num_cond: int = 5
  num_bases: int = 4096
  num_samples: int = 16384
  shard_size: int = 1024

  kernel_variance: float = 0.9  # keep this near 1 since it impacts MC error
  rel_lengthscales_min: float = 0.5
  rel_lengthscales_max: float = 2.0
  num_latent_gps: int = 3

  # Convolutional settings
  channels_in: int = 2
  image_shape: List = [5, 5]
  patch_shape: List = [3, 3]
  strides: List = [1, 1]
  dilations: List = [1, 1]


def _avg_spatial_inner_product(a, b=None, batch_dims: int = 0):
  _a = tf.reshape(a, list(a.shape[:-3]) + [-1, a.shape[-1]])
  if b is None:
    _b = _a
  else:
    _b = tf.reshape(b, list(b.shape[:-3]) + [-1, b.shape[-1]])
  batch_axes = 2 * [list(range(batch_dims))]

  prod = batch_tensordot(_a, _b, axes=[-1, -1], batch_axes=batch_axes)
  return tf.reduce_mean(prod, [-3, -1])


def _test_fourier_conv2d_common(config, kern, X, Z):
  # Use closed-form evaluations as ground truth
  Kuu = covariances.Kuu(Z, kern)
  Kfu = covariances.Kfu(Z, kern, X)
  Kff = kern(X, full_cov=True)

  # Test Fourier-feature-based kernel approximator
  basis = fourier_basis(kern, num_bases=config.num_bases)
  feat_x = basis(X)  # [N, B] or [N, L, B]
  feat_z = basis(Z)

  tol = 3 * config.num_bases ** -0.5
  assert allclose(_avg_spatial_inner_product(feat_x, feat_x), Kff, tol, tol)
  assert allclose(_avg_spatial_inner_product(feat_x, feat_z), Kfu, tol, tol)
  assert allclose(_avg_spatial_inner_product(feat_z, feat_z), Kuu, tol, tol)
  del feat_x, feat_z

  # Test covariance of functions draw from approximate prior
  fx = []
  fz = []
  count = 0
  while count < config.num_samples:
    size = min(config.shard_size, config.num_samples - count)
    funcs = random_fourier(kern,
                           basis=basis,
                           num_bases=config.num_bases,
                           sample_shape=[size])

    fx.append(funcs(X))
    fz.append(funcs(Z))
    count += size

  fx = swap_axes(tf.concat(fx, axis=0), 0, -1)  # [L, N, H, W, S]
  fz = swap_axes(tf.concat(fz, axis=0), 0, -1)  # [L, M, 1, 1, S]
  nb = fx.shape.ndims - 4  # num. of batch dimensions
  tol += 3 * config.num_samples ** -0.5
  frac = 1 / config.num_samples

  assert allclose(frac * _avg_spatial_inner_product(fx, fx, nb), Kff, tol, tol)
  assert allclose(frac * _avg_spatial_inner_product(fx, fz, nb), Kfu, tol, tol)
  assert allclose(frac * _avg_spatial_inner_product(fz, fz, nb), Kuu, tol, tol)


def test_conv2d(config: ConfigFourierConv2d = None):
  """
  TODO: Consider separating out the test for Conv2dTranspose since it only
  supports a subset of strides/dilatons.
  """
  if config is None:
    config = ConfigFourierConv2d()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X_shape = [config.num_test] + config.image_shape + [config.channels_in]
  X = tf.reshape(tf.range(tf.reduce_prod(X_shape), dtype=floatx()), X_shape)
  X /= tf.reduce_max(X)

  Z_shape = [config.num_cond] + config.patch_shape + [config.channels_in]
  Zsrc = tf.random.normal(Z_shape, dtype=floatx())
  Z = inducing_variables.InducingImages(Zsrc)

  patch_len = config.channels_in * config.patch_shape[0] * config.patch_shape[1]
  for base_cls in SupportedBaseKernels:
    minval = config.rel_lengthscales_min * (patch_len ** 0.5)
    maxval = config.rel_lengthscales_max * (patch_len ** 0.5)
    lenscales = tf.random.uniform(shape=[patch_len],
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=floatx())

    base = base_cls(lengthscales=lenscales, variance=config.kernel_variance)
    for cls in (kernels.Conv2d, kernels.Conv2dTranspose):
      kern = cls(kernel=base,
                 image_shape=config.image_shape,
                 patch_shape=config.patch_shape,
                 channels_in=config.channels_in,
                 channels_out=config.num_latent_gps,
                 dilations=config.dilations,
                 strides=config.strides)

      _test_fourier_conv2d_common(config, kern, X, Z)


def test_depthwise_conv2d(config: ConfigFourierConv2d = None):
  if config is None:
    config = ConfigFourierConv2d()

  assert config.num_bases % config.channels_in == 0
  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X_shape = [config.num_test] + config.image_shape + [config.channels_in]
  X = tf.random.uniform(X_shape, dtype=floatx())

  img_shape = [config.num_cond] + config.patch_shape + [config.channels_in]
  Zsrc = tf.random.normal(img_shape, dtype=floatx())
  Z = inducing_variables.DepthwiseInducingImages(Zsrc)

  patch_len = config.patch_shape[0] * config.patch_shape[1]
  for base_cls in SupportedBaseKernels:
    minval = config.rel_lengthscales_min * (patch_len ** 0.5)
    maxval = config.rel_lengthscales_max * (patch_len ** 0.5)
    lenscales = tf.random.uniform(shape=[config.channels_in, patch_len],
                                  minval=minval,
                                  maxval=maxval,
                                  dtype=floatx())

    base = base_cls(lengthscales=lenscales, variance=config.kernel_variance)
    for cls in (kernels.DepthwiseConv2d,):
      kern = cls(kernel=base,
                 image_shape=config.image_shape,
                 patch_shape=config.patch_shape,
                 channels_in=config.channels_in,
                 channels_out=config.num_latent_gps,
                 dilations=config.dilations,
                 strides=config.strides)

      _test_fourier_conv2d_common(config, kern, X, Z)
