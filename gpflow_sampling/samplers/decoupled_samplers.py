#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
import gpflow

from typing import *
from gpflow.config import default_float, default_jitter
from gpflow_sampling.samplers.base import *
from gpflow_sampling.samplers.dispatch import decoupled
from gpflow_sampling.utils import KernelBasis, \
                                  RandomFourierBasis, \
                                  parallel_solve,\
                                  slice_multioutput_kernel,\
                                  slice_multioutput_inducing

# ---- Exports
__all__ = tuple()


# ==============================================
#                             decoupled_samplers
# ==============================================
@decoupled.register(gpflow.models.GPR, gpflow.kernels.Stationary)
def _decoupled_sampler_gpr(model: gpflow.models.GPR,
                           kernel: gpflow.kernels.Stationary,
                           sample_shape: List[int],
                           num_basis: int,
                           prior_basis: Callable = None,
                           dtype: Any = None):
  if dtype is None:
    dtype = default_float()

  def _create_prior_fn(batch_shape, basis: Callable = None):
    if basis is None:
      basis = RandomFourierBasis(kernel=model.kernel,
                                 units=num_basis,
                                 dtype=dtype)

    def w_init(shape, dtype=dtype):
      return tf.random.normal(shape=shape, dtype=dtype)

    weights = tf.Variable(w_init(batch_shape + [num_basis]), trainable=False)
    return BayesianLinearSampler(basis=basis,
                                 weights=weights,
                                 weight_initializer=w_init)

  def _create_update_fn(batch_shape, prior_fn):
    Z, u = model.data
    sigma2 = model.likelihood.variance + default_jitter()
    if model.mean_function is not None:
      u = u - model.mean_function(Z)

    m = Z.shape[-2]
    Kuu = model.kernel(Z, full_cov=True)
    Suu = tf.linalg.set_diag(Kuu, tf.linalg.diag_part(Kuu) + sigma2)
    Luu = tf.linalg.cholesky(Suu)
    basis = KernelBasis(kernel=model.kernel, centers=Z)

    def w_init(shape, dtype=dtype):
      prior_f = prior_fn(Z)
      prior_u = prior_f + (sigma2 ** 0.5) * \
                  tf.random.normal(prior_f.shape, dtype=prior_f.dtype)

      init = tf.linalg.adjoint(parallel_solve(solver=tf.linalg.cholesky_solve,
                                              lhs=Luu,
                                              rhs=u - prior_u))
      assert tuple(init.shape) == tuple(shape)
      return tf.cast(init, dtype)

    weights = tf.Variable(w_init(shape=batch_shape + [m]), trainable=False)
    return BayesianLinearSampler(basis=basis,
                                 weights=weights,
                                 weight_initializer=w_init)

  batch_shape = (sample_shape) + [1]
  prior_fn = _create_prior_fn(batch_shape, prior_basis)
  update_fn = _create_update_fn(batch_shape, prior_fn)
  return CompositeSampler(join_rule=sum,
                          samplers=[prior_fn, update_fn],
                          mean_function=model.mean_function)


@decoupled.register(gpflow.models.SVGP, gpflow.kernels.Stationary)
def _decoupled_sampler_svgp(model: gpflow.models.SVGP,
                            kernel: gpflow.kernels.Stationary,
                            sample_shape: List[int],
                            num_basis: int,
                            prior_basis: Callable = None,
                            latent_dim: int = 0,
                            dtype: Any = None):
  if dtype is None:
    dtype = default_float()

  latent_kernel = slice_multioutput_kernel(model.kernel, latent_dim)
  def _create_prior_fn(batch_shape, basis: Callable = None):
    if basis is None:
      basis = RandomFourierBasis(kernel=latent_kernel,
                                 units=num_basis,
                                 dtype=dtype)

    def w_init(shape, dtype=dtype):
      return tf.random.normal(shape=shape, dtype=dtype)

    weights = tf.Variable(w_init(batch_shape + [num_basis]), trainable=False)
    return BayesianLinearSampler(basis=basis,
                                 weights=weights,
                                 weight_initializer=w_init)

  def _create_update_fn(batch_shape, prior_fn):
    Z = slice_multioutput_inducing(model.inducing_variable, latent_dim)
    Suu = gpflow.covariances.Kuu(Z, latent_kernel, jitter=default_jitter())
    Luu = tf.linalg.cholesky(Suu)
    basis = KernelBasis(kernel=latent_kernel, centers=Z)

    def w_init(shape, dtype=dtype):
      # Sample $u ~ N(f(Z), \epsilon)$ from the prior
      prior_f = prior_fn(Z.Z)  # [!] improve me
      prior_u = prior_f + (default_jitter() ** 0.5) * \
                  tf.random.normal(prior_f.shape, dtype=prior_f.dtype)

      # Sample $u ~ N(q_mu, q_sqrt)$ from inducing distribution $q(u)$
      q_mu = model.q_mu[:, latent_dim: latent_dim + 1]  # Mx1
      q_sqrt = model.q_sqrt[latent_dim]  # MxM
      rvs = tf.random.normal(shape=shape, dtype=dtype)
      induced_u = q_mu + tf.matmul(q_sqrt, rvs, transpose_b=True)
      if model.whiten:
        induced_u = Luu @ induced_u

      init = tf.linalg.adjoint(parallel_solve(solver=tf.linalg.cholesky_solve,
                                              lhs=Luu,
                                              rhs=induced_u - prior_u))

      assert tuple(init.shape) == tuple(shape)
      return tf.cast(init, dtype)

    weights = tf.Variable(w_init(shape=batch_shape + [len(Z)]), trainable=False)
    return BayesianLinearSampler(basis=basis,
                                 weights=weights,
                                 weight_initializer=w_init)

  batch_shape = list(sample_shape) + [1]
  prior_fn = _create_prior_fn(batch_shape, prior_basis)
  update_fn = _create_update_fn(batch_shape, prior_fn)
  return CompositeSampler(join_rule=sum,
                          samplers=[prior_fn, update_fn],
                          mean_function=model.mean_function)
