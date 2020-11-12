#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf

# ---- Exports
__all__ = (
  'normalize_axis',
  'swap_axes',
  'move_axis',
  'expand_n',
  'expand_to',
)


# ==============================================
#                                           misc
# ==============================================
def normalize_axis(axis, ndims, minval=None, maxval=None):
  if minval is None:
    minval = -ndims

  if maxval is None:
    maxval = ndims - 1

  assert maxval >= axis >= minval
  return ndims + axis if (axis < 0) else axis


def move_axis(arr: tf.Tensor, src: int, dest: int):
  ndims = len(arr.shape)
  src = ndims + src if (src < 0) else src
  dest = ndims + dest if (dest < 0) else dest

  src = normalize_axis(src, ndims)
  dest = normalize_axis(dest, ndims)

  perm = list(range(ndims))
  perm.insert(dest, perm.pop(src))
  return tf.transpose(arr, perm)


def swap_axes(arr: tf.Tensor, a: int, b: int):
  ndims = len(arr.shape)
  a = normalize_axis(a, ndims)
  b = normalize_axis(b, ndims)

  perm = list(range(ndims))
  perm[a] = b
  perm[b] = a
  return tf.transpose(arr, perm)


def expand_n(arr: tf.Tensor, axis: int, n: int):
  ndims = len(arr.shape)
  axis = normalize_axis(axis, ndims, maxval=ndims)
  return arr[axis * (slice(None),) + n * (np.newaxis,)]


def expand_to(arr: tf.Tensor, axis: int, ndims: int):
  _ndims = len(arr.shape)
  if _ndims == ndims:
    return tf.identity(arr)
  assert ndims > _ndims
  return expand_n(arr, axis, ndims - _ndims)
