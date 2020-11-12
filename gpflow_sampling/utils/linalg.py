#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf
from typing import List, Union
from string import ascii_lowercase
from collections.abc import Iterable
from gpflow.config import default_jitter
from gpflow_sampling.utils.array_ops import normalize_axis
from tensorflow_probability.python.math import pivoted_cholesky

# ---- Exports
__all__ = (
  'batch_tensordot',
  'get_default_preconditioner',
)


# ==============================================
#                                         linalg
# ==============================================
def batch_tensordot(a: tf.Tensor,
                    b: tf.Tensor,
                    axes: List,
                    batch_axes: List = None) -> tf.Tensor:
  """
  Computes tensor products like <tf.tensordot> with:
    - Support for batch dimensions; 1-to-1 rather than Cartesian product
    - Broadcasting of batch dimensions

  Example:
    a = tf.random.rand([5, 4, 3, 2])
    b = tf.random.rand([6, 4, 2, 1])
    c = batch_tensordot(a, b, [-1, -2], [[-3, -2], [-3, -1]])  # [5, 6, 4, 3]
  """
  ndims_a = len(a.shape)
  ndims_b = len(b.shape)
  assert min(ndims_a, ndims_b) > 0
  assert max(ndims_a, ndims_b) < 26

  # Prepare batch and contraction axes
  def parse_axes(axes):
    if axes is None:
      return [], []

    assert len(axes) == 2
    axes_a, axes_b = axes

    a_is_int = isinstance(axes_a, int)
    b_is_int = isinstance(axes_b, int)
    assert a_is_int == b_is_int
    if a_is_int:
      return [normalize_axis(axes_a, ndims_a)], \
             [normalize_axis(axes_b, ndims_b)]

    assert isinstance(axes_a, Iterable)
    assert isinstance(axes_b, Iterable)
    axes_a = list(axes_a)
    axes_b = list(axes_b)
    length = len(axes_a)
    assert length == len(axes_b)
    if length == 0:
      return [], []

    axes_a = [normalize_axis(ax, ndims_a) for ax in axes_a]
    axes_b = [normalize_axis(ax, ndims_b) for ax in axes_b]
    return map(list, zip(*sorted(zip(axes_a, axes_b))))  # sort according to a

  reduce_a, reduce_b = parse_axes(axes)  # defines the tensor contraction
  batch_a, batch_b = parse_axes(batch_axes)  # group these together 1-to-1

  # Construct left-hand side einsum conventions
  active_a = reduce_a + batch_a
  active_b = reduce_b + batch_b
  assert len(active_a) == len(set(active_a))  # check for duplicates
  assert len(active_b) == len(set(active_b))

  lhs_a = list(ascii_lowercase[:ndims_a])
  lhs_b = list(ascii_lowercase[ndims_a: ndims_a + ndims_b])
  for (pos_a, pos_b) in zip(active_a, active_b):
    lhs_b[pos_b] = lhs_a[pos_a]

  # Construct right-hand side einsum convention
  rhs = []
  for i, char_a in enumerate(lhs_a):
    if i not in reduce_a:
      rhs.append(char_a)

  for i, char_b in enumerate(lhs_b):
    if i not in active_b:
      rhs.append(char_b)

  # Enable broadcasting by eliminate singleton dimenisions
  for (pos_a, pos_b) in zip(batch_a, batch_b):
    if a.shape[pos_a] == b.shape[pos_b]:
      continue  # TODO: test for edge cases

    if a.shape[pos_a] == 1:
      a = tf.squeeze(a, axis=pos_a)
      del lhs_a[pos_a]

    if b.shape[pos_b] == 1:
      b = tf.squeeze(b, axis=pos_b)
      del lhs_b[pos_b]

  # Compute einsum
  return tf.einsum(f"{''.join(lhs_a)},{''.join(lhs_b)}->{''.join(rhs)}", a, b)


def get_default_preconditioner(arr: tf.Tensor,
                               diag: Union[tf.Tensor, tf.linalg.LinearOperator],
                               max_rank: int = 100,
                               diag_rtol: float = None):
  """
  Returns a <tf.linalg.LinearOperator> preconditioner representing
  $(D + LL^{T})^{-1}$ where $D$ is a diagonal matrix and $L$ is the
  partial pivoted Cholesky factor of a symmetric PSD matrix $A$.
  """
  if diag_rtol is None:
    diag_rtol = default_jitter()

  N = arr.shape[-1]
  if not isinstance(diag, tf.linalg.LinearOperator):
    diag = tf.convert_to_tensor(diag, dtype=arr.dtype)
    if N == 1 or (diag.shape.ndims and diag.shape[-1] > 1):
      diag = tf.linalg.LinearOperatorDiag(diag)
    else:
      diag = tf.linalg.LinearOperatorScaledIdentity(N, diag)

  piv_chol = pivoted_cholesky(arr, max_rank=max_rank, diag_rtol=diag_rtol)
  low_rank = tf.linalg.LinearOperatorLowRankUpdate(diag, piv_chol)
  return low_rank.inverse()
