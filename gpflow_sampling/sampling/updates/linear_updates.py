#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
"""
Pathwise updates for Gaussian processes with linear
kernels in some explicit, finite-dimensional basis.
"""
# ---- Imports
import tensorflow as tf

from gpflow import inducing_variables
from gpflow.base import TensorLike
from gpflow.config import default_jitter
from gpflow.utilities import Dispatcher
from gpflow_sampling.utils import swap_axes, move_axis, inducing_to_tensor
from gpflow_sampling.bases.core import AbstractBasis
from gpflow_sampling.sampling.core import DenseSampler, MultioutputDenseSampler


# ==============================================
#                                  linear_update
# ==============================================
linear = Dispatcher("linear_updates")


@linear.register(TensorLike, TensorLike, TensorLike)
def _linear_fallback(Z: TensorLike,
                     u: TensorLike,
                     f: TensorLike,
                     *,
                     L : TensorLike = None,
                     diag: TensorLike = None,
                     basis: AbstractBasis = None,
                     **kwargs):

  u_shape = tuple(u.shape)
  f_shape = tuple(f.shape)
  assert u_shape[-1] == 1, "Recieved multiple output features"
  assert u_shape == f_shape[-len(u_shape):],  "Incompatible shapes detected"

  # Prepare diagonal term
  if diag is None:  # used by <GPflow.conditionals>
    diag = default_jitter()
  if isinstance(diag, float):
    diag = tf.convert_to_tensor(diag, dtype=f.dtype)
  diag = tf.expand_dims(diag, axis=-1)  # [M, 1] or [1, 1] or [1]

  # Extract "features" of Z
  if basis is None:
    if isinstance(Z, inducing_variables.InducingVariables):
      feat = inducing_to_tensor(Z)  # [M, D]
    else:
      feat = Z
  else:
    feat = basis(Z)  # [M, D] (maybe a different "D" than above)

  # Compute error term and matrix square root $Cov(u, u)^{1/2}$
  err = swap_axes(u - f, -3, -1)  # [1, M, S]
  err -= tf.sqrt(diag) * tf.random.normal(err.shape, dtype=err.dtype)
  M, D = feat.shape[-2:]
  if L is None:
    if D < M:
      feat_iDiag = feat * tf.math.reciprocal(diag)
      S = tf.matmul(feat_iDiag, feat, transpose_a=True)  # [D, D]
      L = tf.linalg.cholesky(S + tf.eye(S.shape[-1], dtype=S.dtype))
    else:
      K = tf.matmul(feat, feat, transpose_b=True)  # [M, M]
      K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + diag[..., 0])
      L = tf.linalg.cholesky(K)
  else:
    assert L.shape[-1] == min(M, D)  # TODO: improve me

  # Solve for $Cov(u, u)^{-1}(u - f(Z))$
  if D < M:
    feat_iDiag = feat * tf.math.reciprocal(diag)
    weights = tf.linalg.adjoint(tf.linalg.cholesky_solve(L,
                                tf.matmul(feat_iDiag, err, transpose_a=True)))
  else:
    iK_err = tf.linalg.cholesky_solve(L, err)  # [S, M, 1]
    weights = tf.matmul(iK_err, feat, transpose_a=True)  # [S, 1, D]

  return DenseSampler(basis=basis, weights=move_axis(weights, -2, -3), **kwargs)


@linear.register(inducing_variables.MultioutputInducingVariables,
                 TensorLike,
                 TensorLike)
def _linear_multioutput(Z: inducing_variables.MultioutputInducingVariables,
                        u: TensorLike,
                        f: TensorLike,
                        *,
                        L: TensorLike = None,
                        diag: TensorLike = None,
                        basis: AbstractBasis = None,
                        multioutput_axis: int = "default",
                        **kwargs):
  assert tuple(u.shape) == tuple(f.shape)
  if multioutput_axis == "default":
    multioutput_axis = None if (basis is None) else 0

  # Prepare diagonal term
  if diag is None:  # used by <GPflow.conditionals>
    diag = default_jitter()
  if isinstance(diag, float):
    diag = tf.convert_to_tensor(diag, dtype=f.dtype)
  diag = tf.expand_dims(diag, axis=-1)  # ([L] or []) + ([M] or []) + [1]

  # Extract "features" of Z
  if basis is None:
    if isinstance(Z, inducing_variables.InducingVariables):
      feat = inducing_to_tensor(Z)  # [L, M, D] or [M, D]
    else:
      feat = Z
  elif isinstance(Z, inducing_variables.SharedIndependentInducingVariables):
    feat = basis(Z)
  else:
    feat = basis(Z, multioutput_axis=0)  # first axis of Z is output-specific

  # Compute error term and matrix square root $Cov(u, u)^{1/2}$
  err = swap_axes(u - f, -3, -1)  # [L, M, S]
  err -= tf.sqrt(diag) * tf.random.normal(err.shape, dtype=err.dtype)
  M, D = feat.shape[-2:]
  if L is None:
    if D < M:
      feat_iDiag = feat * tf.math.reciprocal(diag)
      S = tf.matmul(feat_iDiag, feat, transpose_a=True)  # [L, D, D] or [D, D]
      L = tf.linalg.cholesky(S + tf.eye(S.shape[-1], dtype=S.dtype))
    else:
      K = tf.matmul(feat, feat, transpose_b=True)  # [L, M, M] or [M, M]
      K = tf.linalg.set_diag(K, tf.linalg.diag_part(K) + diag[..., 0])
      L = tf.linalg.cholesky(K)
  else:
    assert L.shape[-1] == min(M, D)  # TODO: improve me

  # Solve for $Cov(u, u)^{-1}(u - f(Z))$
  if D < M:
    feat_iDiag = feat * tf.math.reciprocal(diag)
    weights = tf.linalg.adjoint(tf.linalg.cholesky_solve(L,
                                tf.matmul(feat_iDiag, err, transpose_a=True)))
  else:
    iK_err = tf.linalg.cholesky_solve(L, err)  # [L, S, M]
    weights = tf.matmul(iK_err, feat, transpose_a=True)  # [L, S, D]

  return MultioutputDenseSampler(basis=basis,
                                 weights=swap_axes(weights, -3, -2),  # [S, L, D]
                                 multioutput_axis=multioutput_axis,
                                 **kwargs)
