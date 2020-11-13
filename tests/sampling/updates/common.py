#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from numpy import allclose
from typing import Union, NamedTuple
from functools import partial, update_wrapper

from gpflow import config as gpflow_config
from gpflow import kernels, mean_functions
from gpflow.base import TensorLike
from gpflow.config import default_jitter, default_float as floatx
from gpflow.models import GPR, SVGP
from gpflow.kernels import MultioutputKernel, SharedIndependent
from gpflow.utilities import Dispatcher
from gpflow.inducing_variables import (InducingPoints,
                                       InducingVariables,
                                       SharedIndependentInducingVariables,
                                       SeparateIndependentInducingVariables)
from gpflow_sampling import covariances, kernels as kernels_ext
from gpflow_sampling.utils import batch_tensordot
from gpflow_sampling.inducing_variables import *

SupportedBaseKernels = (kernels.Matern12,
                        kernels.Matern32,
                        kernels.Matern52,
                        kernels.SquaredExponential)

# ---- Export
__all__ = ('sample_joint',
           'avg_spatial_inner_product',
           'test_update_sparse',
           'test_update_sparse_shared',
           'test_update_sparse_separate',
           'test_update_conv2d')

# ==============================================
#                                         common
# ==============================================
sample_joint = Dispatcher("sample_joint")


@sample_joint.register(kernels.Kernel, TensorLike, TensorLike)
def _sample_joint_fallback(kern,
                           X,
                           Xnew,
                           num_samples: int,
                           L: TensorLike = None,
                           diag: TensorLike = None):
  """
  Sample from the joint distribution of $f(X), g(Z)$ via a
  location-scale transform.
  """
  if diag is None:
    diag = default_jitter()

  if L is None:
    K = kern(tf.concat([X, Xnew], axis=-2), full_cov=True)
    K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + diag)
    L = tf.linalg.cholesky(K)

  # Draw samples using a location-scale transform
  rvs = tf.random.normal(list(L.shape[:-1]) + [num_samples], dtype=floatx())
  draws = tf.expand_dims(L @ rvs, 0)  # [1, N + T, S]
  return tf.split(tf.transpose(draws), [-1, Xnew.shape[0]], axis=-2), L


@sample_joint.register(kernels.Kernel, InducingVariables, TensorLike)
def _sample_joint_inducing(kern,
                           Z,
                           Xnew,
                           num_samples: int,
                           L: TensorLike = None,
                           diag: Union[float, tf.Tensor] = None):
  """
  Sample from the joint distribution of $f(X), g(Z)$ via a
  location-scale transform.
  """
  if diag is None:
    diag = default_jitter()

  # Construct joint covariance and compute matrix square root
  has_multiple_outputs = isinstance(kern, MultioutputKernel)
  if L is None:
    if has_multiple_outputs:
      Kff = kern(Xnew, full_cov=True, full_output_cov=False)
    else:
      Kff = kern(Xnew, full_cov=True)
    Kuu = covariances.Kuu(Z, kern, jitter=0.0)
    Kuf = covariances.Kuf(Z, kern, Xnew)
    if isinstance(kern, SharedIndependent) and \
       isinstance(Z, SharedIndependentInducingVariables):
      Kuu = tf.tile(Kuu[None], [Kff.shape[0], 1, 1])
      Kuf = tf.tile(Kuf[None], [Kff.shape[0], 1, 1])

    K = tf.concat([tf.concat([Kuu, Kuf], axis=-1),
                   tf.concat([tf.linalg.adjoint(Kuf), Kff], axis=-1)], axis=-2)

    K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + diag)
    L = tf.linalg.cholesky(K)

  # Draw samples using a location-scale transform
  rvs = tf.random.normal(list(L.shape[:-1]) + [num_samples], dtype=floatx())
  draws = L @ rvs  # [L, M + N, S] or [M + N, S]
  if not has_multiple_outputs:
    draws = tf.expand_dims(draws, 0)

  return tf.split(tf.transpose(draws), [-1, Xnew.shape[0]], axis=-2), L


@sample_joint.register(kernels_ext.Conv2d, InducingImages, TensorLike)
def _sample_joint_conv2d(kern,
                         Z,
                         Xnew,
                         num_samples: int,
                         L: TensorLike = None,
                         diag: Union[float, tf.Tensor] = None):
  """
  Sample from the joint distribution of $f(X), g(Z)$ via a
  location-scale transform.
  """
  if diag is None:
    diag = default_jitter()

  # Construct joint covariance and compute matrix square root
  if L is None:
    Zp = Z.as_patches  # [M, patch_len]
    Xp = kern.get_patches(Xnew, full_spatial=False)
    P = tf.concat([Zp, tf.reshape(Xp, [-1, Xp.shape[-1]])], axis=0)
    K = kern.kernel(P, full_cov=True)
    K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + diag)
    L = tf.linalg.cholesky(K)
    L = tf.tile(L[None], [kern.channels_out, 1, 1])  # TODO: Improve me

  # Draw samples using a location-scale transform
  spatial_in = Xnew.shape[-3:-1]
  spatial_out = kern.get_spatial_out(spatial_in)
  rvs = tf.random.normal(list(L.shape[:-1]) + [num_samples], dtype=floatx())
  draws = tf.transpose(L @ rvs)  # [S, M + P, L]
  fz, fx = tf.split(draws, [len(Z), -1], axis=1)

  # Reorganize $f(X)$ as a 3d feature map
  fx_shape = [num_samples, Xnew.shape[0]] + spatial_out + [kern.channels_out]
  fx = tf.reshape(fx, fx_shape)
  return (fz, fx), L


def avg_spatial_inner_product(a, b=None, batch_dims: int = 0):
  """
  Used to compute covariances of functions defined as sums over
  patch response functions in 4D image format [N, H, W, C]
  """
  _a = tf.reshape(a, list(a.shape[:-3]) + [-1, a.shape[-1]])
  if b is None:
    _b = _a
  else:
    _b = tf.reshape(b, list(b.shape[:-3]) + [-1, b.shape[-1]])
  batch_axes = 2 * [list(range(batch_dims))]
  prod = batch_tensordot(_a, _b, axes=[-1, -1], batch_axes=batch_axes)
  return tf.reduce_mean(prod, [-3, -1])


def test_update_dense(default_config: NamedTuple = None):
  def decorator(subroutine):
    def main(config):
      assert config is not None, ValueError
      tf.random.set_seed(config.seed)
      gpflow_config.set_default_float(config.floatx)
      gpflow_config.set_default_jitter(config.jitter)

      X = tf.random.uniform([config.num_cond, config.input_dims], dtype=floatx())
      Xnew = tf.random.uniform([config.num_test, config.input_dims], dtype=floatx())
      for cls in SupportedBaseKernels:
        minval = config.rel_lengthscales_min * (config.input_dims ** 0.5)
        maxval = config.rel_lengthscales_max * (config.input_dims ** 0.5)
        lenscales = tf.random.uniform(shape=[config.input_dims],
                                      minval=minval,
                                      maxval=maxval,
                                      dtype=floatx())

        kern = cls(lengthscales=lenscales, variance=config.kernel_variance)
        const = tf.random.normal([1], dtype=floatx())

        K = kern(X, full_cov=True)
        K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + config.noise_variance)
        L = tf.linalg.cholesky(K)
        y = L @ tf.random.normal([L.shape[-1], 1], dtype=floatx()) + const

        model = GPR(kernel=kern,
                    noise_variance=config.noise_variance,
                    data=(X, y),
                    mean_function=mean_functions.Constant(c=const))

        mf, Sff = subroutine(config, model, Xnew)
        mg, Sgg = model.predict_f(Xnew, full_cov=True)

        tol = config.error_tol
        assert allclose(mf, mg, tol, tol)
        assert allclose(Sff, Sgg, tol, tol)

    return update_wrapper(partial(main, config=default_config), subroutine)
  return decorator


def test_update_sparse(default_config: NamedTuple = None):
  def decorator(subroutine):
    def main(config):
      assert config is not None, ValueError
      tf.random.set_seed(config.seed)
      gpflow_config.set_default_float(config.floatx)
      gpflow_config.set_default_jitter(config.jitter)

      X = tf.random.uniform([config.num_test,config.input_dims], dtype=floatx())
      Z_shape = config.num_cond, config.input_dims
      for cls in SupportedBaseKernels:
        minval = config.rel_lengthscales_min * (config.input_dims ** 0.5)
        maxval = config.rel_lengthscales_max * (config.input_dims ** 0.5)
        lenscales = tf.random.uniform(shape=[config.input_dims],
                                      minval=minval,
                                      maxval=maxval,
                                      dtype=floatx())

        q_sqrt = tf.zeros([1] + 2 * [config.num_cond], dtype=floatx())
        kern = cls(lengthscales=lenscales, variance=config.kernel_variance)
        Z = InducingPoints(tf.random.uniform(Z_shape, dtype=floatx()))

        const = tf.random.normal([1], dtype=floatx())
        model = SVGP(kernel=kern,
                     likelihood=None,
                     inducing_variable=Z,
                     mean_function=mean_functions.Constant(c=const),
                     q_sqrt=q_sqrt)

        mf, Sff = subroutine(config, model, X)
        mg, Sgg = model.predict_f(X, full_cov=True)

        tol = config.error_tol
        assert allclose(mf, mg, tol, tol)
        assert allclose(Sff, Sgg, tol, tol)

    return update_wrapper(partial(main, config=default_config), subroutine)
  return decorator


def test_update_sparse_shared(default_config: NamedTuple = None):
  def decorator(subroutine):
    def main(config):
      assert config is not None, ValueError
      tf.random.set_seed(config.seed)
      gpflow_config.set_default_float(config.floatx)
      gpflow_config.set_default_jitter(config.jitter)

      X = tf.random.uniform([config.num_test,config.input_dims], dtype=floatx())
      Z_shape = config.num_cond, config.input_dims
      for cls in SupportedBaseKernels:
        minval = config.rel_lengthscales_min * (config.input_dims ** 0.5)
        maxval = config.rel_lengthscales_max * (config.input_dims ** 0.5)
        lenscales = tf.random.uniform(shape=[config.input_dims],
                                      minval=minval,
                                      maxval=maxval,
                                      dtype=floatx())

        base = cls(lengthscales=lenscales, variance=config.kernel_variance)
        kern = kernels.SharedIndependent(base, output_dim=2)

        Z = SharedIndependentInducingVariables(
                InducingPoints(tf.random.uniform(Z_shape, dtype=floatx())))
        Kuu = covariances.Kuu(Z, kern, jitter=gpflow_config.default_jitter())
        q_sqrt = tf.stack([tf.zeros(2 * [config.num_cond], dtype=floatx()),
                           tf.linalg.cholesky(Kuu)])

        const = tf.random.normal([2], dtype=floatx())
        model = SVGP(kernel=kern,
                     likelihood=None,
                     inducing_variable=Z,
                     mean_function=mean_functions.Constant(c=const),
                     q_sqrt=q_sqrt,
                     whiten=False,
                     num_latent_gps=2)

        mf, Sff = subroutine(config, model, X)
        mg, Sgg = model.predict_f(X, full_cov=True)
        tol = config.error_tol
        assert allclose(mf, mg, tol, tol)
        assert allclose(Sff, Sgg, tol, tol)
    return update_wrapper(partial(main, config=default_config), subroutine)
  return decorator


def test_update_sparse_separate(default_config: NamedTuple = None):
  def decorator(subroutine):
    def main(config):
      assert config is not None, ValueError
      tf.random.set_seed(config.seed)
      gpflow_config.set_default_float(config.floatx)
      gpflow_config.set_default_jitter(config.jitter)

      X = tf.random.uniform([config.num_test,config.input_dims], dtype=floatx())
      allK = []
      allZ = []
      Z_shape = config.num_cond, config.input_dims
      for cls in SupportedBaseKernels:
        minval = config.rel_lengthscales_min * (config.input_dims ** 0.5)
        maxval = config.rel_lengthscales_max * (config.input_dims ** 0.5)
        lenscales = tf.random.uniform(shape=[config.input_dims],
                                      minval=minval,
                                      maxval=maxval,
                                      dtype=floatx())

        rel_variance = tf.random.uniform(shape=[],
                                         minval=0.9,
                                         maxval=1.1,
                                         dtype=floatx())

        allK.append(cls(lengthscales=lenscales,
                        variance=config.kernel_variance * rel_variance))

        allZ.append(InducingPoints(tf.random.uniform(Z_shape, dtype=floatx())))

      kern = kernels.SeparateIndependent(allK)
      Z = SeparateIndependentInducingVariables(allZ)

      Kuu = covariances.Kuu(Z, kern, jitter=gpflow_config.default_jitter())
      q_sqrt = tf.linalg.cholesky(Kuu)\
               * tf.random.uniform(shape=[kern.num_latent_gps, 1, 1],
                                   minval=0.0,
                                   maxval=0.5,
                                   dtype=floatx())

      const = tf.random.normal([len(kern.kernels)], dtype=floatx())
      model = SVGP(kernel=kern,
                   likelihood=None,
                   inducing_variable=Z,
                   mean_function=mean_functions.Constant(c=const),
                   q_sqrt=q_sqrt,
                   whiten=False,
                   num_latent_gps=len(allK))

      mf, Sff = subroutine(config, model, X)
      mg, Sgg = model.predict_f(X, full_cov=True)
      tol = config.error_tol
      assert allclose(mf, mg, tol, tol)
      assert allclose(Sff, Sgg, tol, tol)
    return update_wrapper(partial(main, config=default_config), subroutine)
  return decorator


def test_update_conv2d(default_config: NamedTuple = None):
  def decorator(subroutine):
    def main(config):
      assert config is not None, ValueError
      tf.random.set_seed(config.seed)
      gpflow_config.set_default_float(config.floatx)
      gpflow_config.set_default_jitter(config.jitter)

      X_shape = [config.num_test] + config.image_shape + [config.channels_in]
      X = tf.reshape(tf.range(tf.reduce_prod(X_shape), dtype=floatx()), X_shape)
      X /= tf.reduce_max(X)

      patch_len = config.channels_in * int(tf.reduce_prod(config.patch_shape))
      for base_cls in SupportedBaseKernels:
        minval = config.rel_lengthscales_min * (patch_len ** 0.5)
        maxval = config.rel_lengthscales_max * (patch_len ** 0.5)
        lenscales = tf.random.uniform(shape=[patch_len],
                                      minval=minval,
                                      maxval=maxval,
                                      dtype=floatx())

        base = base_cls(lengthscales=lenscales, variance=config.kernel_variance)
        Z_shape = [config.num_cond] + config.patch_shape + [config.channels_in]
        for cls in (kernels_ext.Conv2d, kernels_ext.Conv2dTranspose):
          kern = cls(kernel=base,
                     image_shape=config.image_shape,
                     patch_shape=config.patch_shape,
                     channels_in=config.channels_in,
                     channels_out=config.num_latent_gps,
                     strides=config.strides,
                     padding=config.padding,
                     dilations=config.dilations)

          Z = InducingImages(tf.random.uniform(Z_shape, dtype=floatx()))
          q_sqrt = tf.linalg.cholesky(covariances.Kuu(Z, kern))
          q_sqrt *= tf.random.uniform([config.num_latent_gps, 1, 1],
                                      minval=0.0,
                                      maxval=0.5,
                                      dtype=floatx())

          # TODO: GPflow's SVGP class is not setup to support outputs defined
          #       as spatial feature maps. For now, we content ourselves with
          #       the following hack...
          const = tf.random.normal([config.num_latent_gps], dtype=floatx())
          mean_function = lambda x: const

          model = SVGP(kernel=kern,
                       likelihood=None,
                       mean_function=mean_function,
                       inducing_variable=Z,
                       q_sqrt=q_sqrt,
                       whiten=False,
                       num_latent_gps=config.num_latent_gps)

          mf, Sff = subroutine(config, model, X)
          mg, Sgg = model.predict_f(X, full_cov=True)

          tol = config.error_tol
          assert allclose(mf, mg, tol, tol)
          assert allclose(Sff, Sgg, tol, tol)

    return update_wrapper(partial(main, config=default_config), subroutine)
  return decorator
