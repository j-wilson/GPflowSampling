#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
import gpflow

from typing import *
from gpflow.base import TensorType
from gpflow.config import default_jitter
from gpflow_sampling.utils.linalg_ops import parallel_solve
from gpflow_sampling.samplers.base import Sampler
from gpflow_sampling.samplers.dispatch import location_scale

# ---- Exports
__all__ = (
  'LocationScaleSamplerGPR',
  'CacheLocationScaleSamplerGPR',
  'LocationScaleSamplerSVGP',
  'CacheLocationScaleSamplerSVGP',
)


# ==============================================
#                        location_scale_samplers
# ==============================================
@location_scale.register(gpflow.models.GPR, gpflow.kernels.Kernel)
def _location_scale_gpr(model: gpflow.models.GPR,
                        kernel: gpflow.kernels.Kernel,
                        sample_shape: List[int] = None,
                        precompute: bool = True):

  cache = LocationScaleSamplerGPR.build_cache(model) if precompute else None
  return LocationScaleSamplerGPR(model=model,
                                 sample_shape=sample_shape,
                                 cache=cache)


@location_scale.register(gpflow.models.SVGP, gpflow.kernels.Kernel)
def _location_scale_svgp(model: gpflow.models.SVGP,
                         kernel: gpflow.kernels.Kernel,
                         sample_shape: List[int],
                         precompute: bool = True):

  cache = LocationScaleSamplerSVGP.build_cache(model) if precompute else None
  return LocationScaleSamplerSVGP(model=model,
                                  sample_shape=sample_shape,
                                  cache=cache)


class CacheLocationScaleSamplerGPR(NamedTuple):
  Z: TensorType
  Luu: TensorType  # Cholesky factor of $Cov(u, u)$
  iLuu_err: TensorType


class LocationScaleSamplerGPR(Sampler):
  def __init__(self,
               model: gpflow.models.GPR,
               sample_shape: List[int],
               full_cov: bool = True,
               cache: CacheLocationScaleSamplerGPR = None):
    self.model = model
    self.sample_shape = sample_shape
    self.full_cov = full_cov
    self._cache = cache

  def __call__(self,
               X: TensorType,
               sample_shape: List[int] = None,
               full_cov: bool = None) -> tf.Tensor:

    if full_cov is None:
       full_cov = self.full_cov

    if sample_shape is None:
      sample_shape = self.sample_shape

    # Get or compute required terms
    Z, Luu, iLuu_err = self.cache
    rvs = tf.random.normal(shape=list(sample_shape) + list(X.shape[:-1]),
                           dtype=X.dtype)

    # Solve for linear systems involving $Cov(u, u)$
    Kuf = tf.linalg.adjoint(self.model.kernel(X, Z))  # for broadcasting
    iLuu_Kuf = parallel_solve(tf.linalg.triangular_solve, Luu, Kuf)

    # Compute and draw samples from posterior
    m = tf.matmul(iLuu_Kuf, iLuu_err, transpose_a=True)
    if self.model.mean_function is not None:
      m += self.model.mean_function(X)

    if full_cov:
      S = self.model.kernel(X, full_cov=True) \
          - tf.matmul(iLuu_Kuf, iLuu_Kuf, transpose_a=True)
      L = tf.linalg.cholesky(
          tf.linalg.set_diag(S, tf.linalg.diag_part(S) + default_jitter()))
      return m + tf.expand_dims(tf.linalg.matvec(L, rvs), -1)

    v = self.model.kernel(X, full_cov=False) \
        - tf.reduce_sum(tf.square(iLuu_Kuf), axis=-2)

    return m + tf.expand_dims(tf.sqrt(v) * rvs, axis=-1)

  def reset_random_variables(self, *args, **kwargs):
    pass

  @classmethod
  def build_cache(cls, model: gpflow.models.GPR):
    Z, err = model.data
    sigma2 = model.likelihood.variance + default_jitter()
    if model.mean_function is not None:
      err -= model.mean_function(Z)

    Kuu = model.kernel(Z, full_cov=True)
    Suu = tf.linalg.set_diag(Kuu, tf.linalg.diag_part(Kuu) + sigma2)
    Luu = tf.linalg.cholesky(Suu)
    iLuu_err = parallel_solve(tf.linalg.triangular_solve, Luu, err)
    return CacheLocationScaleSamplerGPR(Z=Z,
                                        Luu=tf.linalg.cholesky(Suu),
                                        iLuu_err=iLuu_err)

  @property
  def cache(self):
    if self._cache is None:
      return self.build_cache(self.model)
    return self._cache

  @cache.setter
  def cache(self, cache):
    self._cache = cache


class CacheLocationScaleSamplerSVGP(NamedTuple):
  Z: TensorType
  Luu: TensorType  # Cholesky factor of $Cov(u, u)$
  q_mu: TensorType
  q_sqrt: TensorType


class LocationScaleSamplerSVGP(Sampler):
  def __init__(self,
               model: gpflow.models.SVGP,
               sample_shape,
               full_cov: bool = True,
               cache: CacheLocationScaleSamplerSVGP = None):
    self.model = model
    self.sample_shape = sample_shape
    self.full_cov = full_cov
    self._cache = cache

  def __call__(self, X: TensorType,
               sample_shape: List[int] = None,
               full_cov: bool = None) -> tf.Tensor:

    if full_cov is None:
       full_cov = self.full_cov

    if sample_shape is None:
      sample_shape = self.sample_shape

    # Get or compute required terms
    Z, Luu, q_mu, q_sqrt = self.cache
    rvs = tf.random.normal(shape=list(sample_shape) + list(X.shape[:-1]),
                           dtype=X.dtype)

    # Solve for $Cov(u, u)^{-1/2} Cov(u, f)$
    # [!] Fix me: doesn't broadcast in the desired way
    #     Kuf = gpflow.covariances.Kuf(Z, self.model.kernel, X)
    Kuf = tf.linalg.adjoint(self.model.kernel(X, Z.Z))
    iLuu_Kuf = parallel_solve(tf.linalg.triangular_solve, Luu, Kuf)

    # Compute and draw samples from posterior
    if self.model.whiten:
      m = tf.matmul(iLuu_Kuf, q_mu, transpose_a=True)
      A = tf.matmul(iLuu_Kuf, q_sqrt, transpose_a=True)
    else:
      iSuu_Kuf = parallel_solve(tf.linalg.triangular_solve,
                                tf.linalg.adjoint(Luu),
                                iLuu_Kuf,
                                lower=False)
      m = tf.matmul(iSuu_Kuf, q_mu, transpose_a=True)
      A = tf.matmul(iSuu_Kuf, q_sqrt, transpose_a=True)

    if self.model.mean_function is not None:
      m += self.model.mean_function(X)

    if full_cov:
      S = self.model.kernel(X, full_cov=True) \
          + tf.matmul(A, A, transpose_b=True) \
          - tf.matmul(iLuu_Kuf, iLuu_Kuf, transpose_a=True)

      L = tf.linalg.cholesky(
            tf.linalg.set_diag(S, tf.linalg.diag_part(S) + default_jitter()))
      return m + tf.expand_dims(tf.linalg.matvec(L, rvs), -1)

    v = self.model.kernel(X, full_cov=False) \
        + tf.reduce_sum(tf.square(A) - tf.square(iLuu_Kuf), axis=-2)
    return m + tf.expand_dims(tf.sqrt(v) * rvs, axis=-1)

  def reset_random_variables(self, *args, **kwargs):
    pass

  @classmethod
  def build_cache(cls, model: gpflow.models.SVGP):
    assert model.q_sqrt.shape.ndims == 3 and model.q_sqrt.shape[0] == 1
    q_mu = model.q_mu
    q_sqrt = model.q_sqrt[0]

    Z = model.inducing_variable
    Suu = gpflow.covariances.Kuu(Z, model.kernel, jitter=default_jitter())
    return CacheLocationScaleSamplerSVGP(Z=Z,
                                         Luu=tf.linalg.cholesky(Suu),
                                         q_mu=q_mu,
                                         q_sqrt=q_sqrt)

  @property
  def cache(self):
    if self._cache is None:
      return self.build_cache(model=self.model)

    return self._cache

  @cache.setter
  def cache(self, cache):
    self._cache = cache
