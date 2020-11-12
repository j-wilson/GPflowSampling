#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from . import common
from typing import Any, List, Union, NamedTuple
from gpflow.config import default_jitter, default_float as floatx
from gpflow.models import GPR, SVGP
from gpflow_sampling import covariances
from gpflow_sampling.utils import swap_axes
from gpflow_sampling.sampling.updates import exact as exact_update


# ==============================================
#                                     test_exact
# ==============================================
class ConfigDense(NamedTuple):
  seed: int = 0
  floatx: Any = 'float64'
  jitter: float = 1e-6

  num_test: int = 128
  num_cond: int = 32
  num_samples: int = 16384
  shard_size: int = 1024
  input_dims: int = 3

  kernel_variance: float = 0.9  # keep this near 1 since it impacts MC error
  rel_lengthscales_min: float = 0.05
  rel_lengthscales_max: float = 0.5
  noise_variance: float = 1e-2  # only used by GPR test

  @property
  def error_tol(self):
    return 4 * (self.num_samples ** -0.5)


class ConfigConv2d(NamedTuple):
  seed: int = 1
  floatx: Any = 'float64'
  jitter: float = 1e-5

  num_test: int = 128
  num_cond: int = 32
  num_samples: int = 16384
  shard_size: int = 1024
  kernel_variance: float = 0.9
  rel_lengthscales_min: float = 0.5
  rel_lengthscales_max: float = 2.0
  num_latent_gps: int = 3

  # Convolutional settings
  channels_in: int = 2
  image_shape: List = [3, 3]  # Keep these small, since common.sample_joint
  patch_shape: List = [2, 2]  # becomes very expensive for Conv2dTranspose!
  strides: List = [1, 1]
  padding: str = "VALID"
  dilations: List = [1, 1]

  @property
  def error_tol(self):
    return 4 * (self.num_samples ** -0.5)


def _test_exact_gpr(config: ConfigDense,
                    model: GPR,
                    Xnew: tf.Tensor) -> tf.Tensor:
  """
  Sample generation subroutine common to each unit test
  """
  # Precompute Cholesky factor (optional)
  X, y = model.data
  Kyy = model.kernel(X, full_cov=True)
  Kyy = tf.linalg.set_diag(Kyy,
                           tf.linalg.diag_part(Kyy) + model.likelihood.variance)
  Lyy = tf.linalg.cholesky(Kyy)

  count = 0
  L_joint = None
  samples = []
  while count < config.num_samples:
    # Sample $u ~ N(q_mu, q_sqrt q_sqrt^{T})$
    size = min(config.shard_size, config.num_samples - count)

    # Generate draws from the joint distribution $p(f(X), f(Xnew))$
    (f, fnew), L_joint = common.sample_joint(model.kernel,
                                                X,
                                                Xnew,
                                                num_samples=size,
                                                L=L_joint)

    # Solve for update functions
    update_fns = exact_update(model.kernel,
                              X,
                              y,
                              f + model.mean_function(X),
                              L=Lyy,
                              diag=model.likelihood.variance)

    samples.append(fnew + update_fns(Xnew))
    count += size

  samples = tf.concat(samples, axis=0)
  if model.mean_function is not None:
    samples += model.mean_function(Xnew)
  return samples


def _test_exact_svgp(config: Union[ConfigDense, ConfigConv2d],
                     model: SVGP,
                     Xnew: tf.Tensor) -> tf.Tensor:
  """
  Sample generation subroutine common to each unit test
  """
  # Precompute Cholesky factor (optional)
  Z = model.inducing_variable
  Kuu = covariances.Kuu(Z, model.kernel, jitter=default_jitter())
  Luu = tf.linalg.cholesky(Kuu)

  count = 0
  L_joint = None
  samples = []
  while count < config.num_samples:
    # Sample $u ~ N(q_mu, q_sqrt q_sqrt^{T})$
    size = min(config.shard_size, config.num_samples - count)
    shape = model.num_latent_gps, config.num_cond, size
    rvs = tf.random.normal(shape=shape, dtype=floatx())
    u = tf.transpose(model.q_sqrt @ rvs)

    # Generate draws from the joint distribution $p(f(X), g(Z))$
    (f, fnew), L_joint = common.sample_joint(model.kernel,
                                             Z,
                                             Xnew,
                                             num_samples=size,
                                             L=L_joint)

    # Solve for update functions
    update_fns = exact_update(model.kernel, Z, u, f, L=Luu)
    samples.append(fnew + update_fns(Xnew))
    count += size

  samples = tf.concat(samples, axis=0)
  if model.mean_function is not None:
    samples += model.mean_function(Xnew)
  return samples


@common.test_update_dense(default_config=ConfigDense())
def test_exact_dense(*args, **kwargs):
  f = _test_exact_gpr(*args, **kwargs)  # [S, N, 1]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.squeeze(f - mf, axis=-1)
  Sff = tf.matmul(res, res, transpose_a=True)/f.shape[0]
  return mf, Sff


@common.test_update_sparse(default_config=ConfigDense())
def test_exact_sparse(*args, **kwargs):
  f = _test_exact_svgp(*args, **kwargs)  # [S, N, 1]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.squeeze(f - mf, axis=-1)
  Sff = tf.matmul(res, res, transpose_a=True)/f.shape[0]
  return mf, Sff


@common.test_update_sparse_shared(default_config=ConfigDense())
def test_exact_sparse_shared(*args, **kwargs):
  f = _test_exact_svgp(*args, **kwargs)  # [S, N, L]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.transpose(f - mf)
  Sff = tf.matmul(res, res, transpose_b=True)/f.shape[0]
  return mf, Sff


@common.test_update_sparse_separate(default_config=ConfigDense())
def test_exact_sparse_separate(*args, **kwargs):
  f = _test_exact_svgp(*args, **kwargs)  # [S, N, L]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.transpose(f - mf)
  Sff = tf.matmul(res, res, transpose_b=True)/f.shape[0]
  return mf, Sff


@common.test_update_conv2d(default_config=ConfigConv2d())
def test_exact_conv2d(*args, **kwargs):
  f = swap_axes(_test_exact_svgp(*args, **kwargs), 0, -1)  # [L, N, H, W, S]
  mf = tf.transpose(tf.reduce_mean(f, [-3, -2, -1]))  # [N, L]
  res = f - tf.reduce_mean(f, axis=-1, keepdims=True)
  Sff = common.avg_spatial_inner_product(res, batch_dims=f.shape.ndims - 4)
  return mf, Sff/f.shape[-1]  # [L, N, N]
