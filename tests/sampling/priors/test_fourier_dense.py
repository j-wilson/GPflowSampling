#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from numpy import allclose
from typing import Any, NamedTuple
from gpflow import kernels, config as gpflow_config
from gpflow.inducing_variables import (InducingPoints,
                                       MultioutputInducingVariables,
                                       SharedIndependentInducingVariables,
                                       SeparateIndependentInducingVariables)
from gpflow.config import default_float as floatx
from gpflow_sampling import covariances
from gpflow_sampling.bases import fourier as fourier_basis
from gpflow_sampling.sampling.priors import random_fourier

SupportedBaseKernels = (kernels.Matern12,
                        kernels.Matern32,
                        kernels.Matern52,
                        kernels.SquaredExponential)


# ==============================================
#                             test_fourier_dense
# ==============================================
class ConfigFourierDense(NamedTuple):
  seed: int = 1
  floatx: Any = 'float64'
  jitter: float = 1e-6

  num_test: int = 16
  num_cond: int = 8
  num_bases: int = 4096
  num_samples: int = 16384
  shard_size: int = 1024
  input_dims: int = 3

  kernel_variance: float = 0.9  # keep this near 1 since it impacts MC error
  lengthscales_min: float = 0.1
  lengthscales_max: float = 1.0


def _test_fourier_dense_common(config, kern, X, Z):
  # Test Fourier-feature-based kernel approximator
  Kuu = covariances.Kuu(Z, kern)
  Kfu = covariances.Kfu(Z, kern, X)
  basis = fourier_basis(kern, num_bases=config.num_bases)
  Z_opt = dict()  # options used when evaluating basis/prior at Z
  if isinstance(Z, MultioutputInducingVariables):
    Kff = kern(X, full_cov=True, full_output_cov=False)
    if not isinstance(Z, SharedIndependentInducingVariables):
      # Handling for non-shared multioutput inducing variables.
      # We need to indicate that Z's outermost axis should be
      # evaluated 1-to-1 with the L latent GPs
      Z_opt.setdefault("multioutput_axis", 0)

    feat_x = basis(X)  # [L, N, B]
    feat_z = basis(Z, **Z_opt)  # [L, M, B]
  else:
    Kff = kern(X, full_cov=True)
    feat_x = basis(X)
    feat_z = basis(Z)

  tol = 3 * config.num_bases ** -0.5
  assert allclose(tf.matmul(feat_x, feat_x, transpose_b=True), Kff, tol, tol)
  assert allclose(tf.matmul(feat_x, feat_z, transpose_b=True), Kfu, tol, tol)
  assert allclose(tf.matmul(feat_z, feat_z, transpose_b=True), Kuu, tol, tol)
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
    fz.append(funcs(Z, **Z_opt))
    count += size

  fx = tf.transpose(tf.concat(fx, axis=0))  # [L, N, S]
  fz = tf.transpose(tf.concat(fz, axis=0))  # [L, M, S]
  tol += 3 * config.num_samples ** -0.5
  frac = 1 / config.num_samples
  assert allclose(frac * tf.matmul(fx, fx, transpose_b=True), Kff, tol, tol)
  assert allclose(frac * tf.matmul(fx, fz, transpose_b=True), Kfu, tol, tol)
  assert allclose(frac * tf.matmul(fz, fz, transpose_b=True), Kuu, tol, tol)


def test_dense(config: ConfigFourierDense = None):
  if config is None:
    config = ConfigFourierDense()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X = tf.random.uniform([config.num_test, config.input_dims], dtype=floatx())
  Z = tf.random.uniform([config.num_cond, config.input_dims], dtype=floatx())
  Z = InducingPoints(Z)
  for cls in SupportedBaseKernels:
    lenscales = tf.random.uniform(shape=[config.input_dims],
                                  minval=config.lengthscales_min,
                                  maxval=config.lengthscales_max,
                                  dtype=floatx())

    kern = cls(lengthscales=lenscales, variance=config.kernel_variance)
    _test_fourier_dense_common(config, kern, X, Z)


def test_dense_shared(config: ConfigFourierDense = None, output_dim: int = 2):
  if config is None:
    config = ConfigFourierDense()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  X = tf.random.uniform([config.num_test, config.input_dims], dtype=floatx())
  Z = tf.random.uniform([config.num_cond, config.input_dims], dtype=floatx())
  Z = SharedIndependentInducingVariables(InducingPoints(Z))
  for cls in SupportedBaseKernels:
    lenscales = tf.random.uniform(shape=[config.input_dims],
                                  minval=config.lengthscales_min,
                                  maxval=config.lengthscales_max,
                                  dtype=floatx())

    base = cls(lengthscales=lenscales, variance=config.kernel_variance)
    kern = kernels.SharedIndependent(base, output_dim=output_dim)
    _test_fourier_dense_common(config, kern, X, Z)


def test_dense_separate(config: ConfigFourierDense = None):
  if config is None:
    config = ConfigFourierDense()

  tf.random.set_seed(config.seed)
  gpflow_config.set_default_float(config.floatx)
  gpflow_config.set_default_jitter(config.jitter)

  allZ = []
  allK = []
  for cls in SupportedBaseKernels:
    lenscales = tf.random.uniform(shape=[config.input_dims],
                                  minval=config.lengthscales_min,
                                  maxval=config.lengthscales_max,
                                  dtype=floatx())

    rel_variance = tf.random.uniform(shape=[],
                                     minval=0.9,
                                     maxval=1.1,
                                     dtype=floatx())

    allK.append(cls(lengthscales=lenscales,
                    variance=config.kernel_variance * rel_variance))

    allZ.append(InducingPoints(
      tf.random.uniform([config.num_cond, config.input_dims], dtype=floatx())))

  kern = kernels.SeparateIndependent(allK)
  Z = SeparateIndependentInducingVariables(allZ)
  X = tf.random.uniform([config.num_test, config.input_dims], dtype=floatx())
  _test_fourier_dense_common(config, kern, X, Z)


if __name__ == "__main__":
  test_dense()
  test_dense_shared()
  test_dense_separate()
