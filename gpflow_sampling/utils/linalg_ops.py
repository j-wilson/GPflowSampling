#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf

from typing import Callable
from gpflow.config import default_jitter

# ---- Exports
__all__ = (
  'parallel_solve',
  'jitter_diagonal',
  'jitter_cholesky',
  'givens_rotation',  # These methods are
  'cholesky_update',  # implemented in Numpy
)


# ==============================================
#                                     linalg_ops
# ==============================================
def parallel_solve(solver: Callable,
                   lhs: tf.Tensor,
                   rhs: tf.Tensor,
                   **kwargs) -> tf.Tensor:
  """
  Simultaneously solve for one or more linear systems.
  """
  shape_lhs = list(lhs.shape)
  shape_rhs = list(rhs.shape)
  ndims_lhs = len(shape_lhs)
  ndims_rhs = len(shape_rhs)
  rank_diff = ndims_rhs - ndims_lhs  # change in rank

  if rank_diff == 0:
    broadcast = []
    for i, (_lhs, _rhs) in enumerate(zip(shape_lhs[:-2], shape_rhs[:-2])):
      if _lhs != _rhs:
        assert (_lhs == 1) or (_rhs == 1)
      broadcast.append(max(_lhs, _rhs))

    x = solver(tf.broadcast_to(lhs, broadcast + shape_lhs[-2:]),
               tf.broadcast_to(rhs, broadcast + shape_rhs[-2:]),
               **kwargs)

  elif rank_diff > 0:
    _lhs = tf.convert_to_tensor(lhs)  # [!] ugly, strip leading singletons
    while _lhs.shape.ndims > 2 and _lhs.shape[0] == 1:
      _lhs = tf.squeeze(_lhs, axis=0)
      rank_diff += 1

    # Transpose and reshape RHS as a matrix
    axes = list(range(ndims_rhs))
    perm_rhs = axes[rank_diff:-1] + axes[:rank_diff] + axes[-1:]
    shape_vect = shape_rhs[rank_diff:-1] + [-1]
    rhs_vect = tf.reshape(tf.transpose(rhs, perm_rhs), shape_vect)

    # Solve for X matrix, then reshape and transpose
    x_vect = solver(_lhs, rhs_vect, **kwargs)
    shape_nd = shape_rhs[rank_diff:-1] + shape_rhs[:rank_diff] + shape_rhs[-1:]
    perm_x = axes[-(rank_diff + 1): -1] + axes[:-(rank_diff + 1)] + axes[-1:]
    x = tf.transpose(tf.reshape(x_vect, shape_nd), perm_x)
  else:
    _rhs = tf.broadcast_to(tf.convert_to_tensor(rhs)[rank_diff * (np.newaxis,)],
                           shape_lhs[:(-rank_diff)] + shape_rhs)
    x = solver(lhs, _rhs, **kwargs)

  return x


def jitter_diagonal(x, jitter=None, name=None):
  """
  Add jitter to the diagonal of a square tensor of rank >= 2.
  """
  assert x.shape.ndims >= 2, 'Input tensor must be at least 2D'
  if jitter is None:
    jitter = default_jitter()

  tf.assert_equal(x.shape[-1], x.shape[-2], name='test_square')
  return tf.linalg.set_diag(x, tf.linalg.diag_part(x) + jitter)


def jitter_cholesky(x, jitter=None, upcast=False, name=None, **kwargs):
  """
  Cholesky factorization of >=2-dimensional square tensor
  'x', stabilized via additition of a small, positive
  constant to its diagonal.
  """
  dtype = tf.as_dtype(x.dtype)
  if upcast and dtype != tf.float64:
    x = tf.cast(x, tf.float64)

  _x = jitter_diagonal(x, jitter=jitter, **kwargs)
  chol = tf.linalg.cholesky(_x)
  if chol.dtype != dtype:
    chol = tf.cast(chol, dtype, name=name)
  elif name is not None:
    chol = tf.identity(chol, name=name)
  return chol


def givens_rotation(a, b):
  if b == 0:
    c = np.sign(a)
    if c == 0:
      c = 1.0
    s = 0
    r = abs(a)
  elif a == 0:
    c = 0
    s = np.sign(b)
    r = abs(b)
  elif abs(a) > abs(b):
    t = b/a
    u = np.sign(a) * np.sqrt(1 + t*t)
    c = 1/u
    s = c*t
    r = a*u
  else:
    t = a/b
    u = np.sign(b) * np.sqrt(1 + t*t)
    s = 1/u
    c = s*t
    r = b*u
  return c, s, r


def cholesky_update(L, x, sign='+'):
  n = L.shape[-1]
  for i in range(n):
    xi = x[..., i]
    Lii = L[..., i, i]
    if sign == '+':
      c, s, L[..., i, i] = givens_rotation(Lii, xi)
    else:
      s = xi/Lii
      s2 = np.square(s)
      assert np.all(s2 <= 1.0)
      c = np.sqrt(1 - s2)
      L[..., i, i] = c * Lii

    for j in range(i+1, n):
      if sign == '+':
        Lji = L[..., j, i].copy()
        L[..., j, i] = c * Lji + s * x[..., j]
      else:
        Lji = L[..., j, i] = (L[..., j, i] - s * x[..., j])/c
      x[..., j] *= c
      x[..., j] -= s * Lji
  return L