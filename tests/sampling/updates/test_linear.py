#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from . import common
from typing import Any, NamedTuple
from gpflow.config import default_float as floatx
from gpflow.models import SVGP
from gpflow_sampling.bases import fourier as fourier_basis
from gpflow_sampling.sampling.updates import linear as linear_update


# ==============================================
#                                    test_linear
# ==============================================
class UnitTestConfigDense(NamedTuple):
  seed: int = 0
  floatx: Any = 'float64'
  jitter: float = 1e-6

  num_test: int = 128
  num_cond: int = 32
  num_bases: int = 4096
  num_samples: int = 16384
  shard_size: int = 1024
  input_dims: int = 3

  kernel_variance: float = 0.9  # keep this near 1 since it impacts MC error
  rel_lengthscales_min: float = 0.05
  rel_lengthscales_max: float = 0.5

  @property
  def error_tol(self):
    return 4 * (self.num_samples ** -0.5 + self.num_bases ** -0.5)


def _test_linear_svgp(config: UnitTestConfigDense,
                      model: SVGP,
                      Xnew: tf.Tensor) -> tf.Tensor:
  """
  Sample generation subroutine common to each unit test
  """
  Z = model.inducing_variable
  count = 0
  basis = fourier_basis(model.kernel, num_bases=config.num_bases)
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
    update_fns = linear_update(Z, u, f, basis=basis)
    samples.append(fnew + update_fns(Xnew))
    count += size

  samples = tf.concat(samples, axis=0)
  if model.mean_function is not None:
    samples += model.mean_function(Xnew)
  return samples


@common.test_update_sparse(default_config=UnitTestConfigDense())
def test_linear_sparse(*args, **kwargs):
  f = _test_linear_svgp(*args, **kwargs)  # [S, N, 1]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.squeeze(f - mf, axis=-1)
  Sff = tf.matmul(res, res, transpose_a=True)/f.shape[0]
  return mf, Sff


@common.test_update_sparse_shared(default_config=UnitTestConfigDense())
def test_linear_sparse_shared(*args, **kwargs):
  f = _test_linear_svgp(*args, **kwargs)  # [S, N, L]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.transpose(f - mf)
  Sff = tf.matmul(res, res, transpose_b=True)/f.shape[0]
  return mf, Sff


@common.test_update_sparse_separate(default_config=UnitTestConfigDense())
def test_linear_sparse_separate(*args, **kwargs):
  f = _test_linear_svgp(*args, **kwargs)  # [S, N, L]
  mf = tf.reduce_mean(f, axis=0)
  res = tf.transpose(f - mf)
  Sff = tf.matmul(res, res, transpose_b=True)/f.shape[0]
  return mf, Sff
