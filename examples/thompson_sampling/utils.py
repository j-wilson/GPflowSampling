#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf

from typing import *
from gpflow.config import default_float

# ---- Exports
__all__ = (
  'take_along_axis',
  'take_top_k',
  'find_top_k',
  'build_batch_provider',
)


# ==============================================
#                                          utils
# ==============================================
def take_along_axis(arr: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
  """
  Tensorflow equivalent of <numpy.take_along_axis>
  """
  _arr = tf.convert_to_tensor(arr)
  _idx = tf.convert_to_tensor(indices)
  _axis = arr.shape.ndims + axis if (axis < 0) else axis

  components = []
  for i, (size_a, size_i) in enumerate(zip(_arr.shape, _idx.shape)):
    if i == _axis:
      components.append(tf.range(size_i, dtype=_idx.dtype))
    elif size_a == 1:
      components.append(tf.zeros(size_i, dtype=_idx.dtype))
    else:
      assert size_i in (1, size_a), \
        ValueError(f'Shape mismatch: {_arr.shape} vs {_idx.shape}')
      components.append(tf.range(size_a, dtype=_idx.dtype))

  mesh = tf.meshgrid(*components, indexing='ij')
  mesh[_axis] = tf.broadcast_to(_idx, mesh[0].shape)
  indices_nd = tf.stack(mesh, axis=-1)
  return tf.gather_nd(arr, indices_nd)


def take_top_k(arr: tf.Tensor,
               vals: tf.Tensor,
               k: int,
               sign: str = '+',
               **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  Returns the top-k elements from a tensor ranked according
  to a corresponding tensor of values.

  :param arr - Tensor of shape [..., N, D]
  :param vals - Tensor of shape [---, ..., N]
  :param k - Number of top elements to return
  :param: sign - string indicator for min/max

  :return top_arr - Tensor of shape [---, ..., k, D]
  :return top_vals - Tensor of shape [---, ..., k]
  """
  if sign == '-':
    vals = -vals
  else:
    assert sign == '+', ValueError("sign <str> must be '+' or '-'")

  top_vals, i = tf.nn.top_k(vals, k=k, **kwargs)  # always along last axis
  indices = i[..., None]
  newaxes = (indices.shape.ndims - arr.shape.ndims) * (np.newaxis,)
  top_arr = take_along_axis(arr[newaxes], indices, axis=-2)
  if sign == '-':
    top_vals = -top_vals

  return top_arr, top_vals


def find_top_k(fn: Callable[[tf.Tensor], tf.Tensor],
               batch_provider: Generator,
               k: int,
               **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
  """
  Returns a function's top-k extremal points (and values) defined
  over a series of batches.
  """
  for i, xvals, in enumerate(map(tf.convert_to_tensor, batch_provider())):
    fvals = fn(xvals)
    shape = list(fvals.shape[:1 - xvals.shape.ndims]) + list(xvals.shape)
    xvals = tf.broadcast_to(xvals, shape)
    if i == 0:
      best_xvals = xvals
      best_fvals = fvals
    else:
      best_xvals = tf.concat([best_xvals, xvals], axis=-2)
      best_fvals = tf.concat([best_fvals, fvals], axis=-1)

    if best_xvals.shape[-2] > k:
      best_xvals, best_fvals = take_top_k(arr=best_xvals,
                                          vals=best_fvals,
                                          k=k,
                                          **kwargs)
  return best_xvals, best_fvals


def build_batch_provider(num_batches: int,
                         batch_shape: List[int],
                         subroutine: Callable = tf.random.uniform,
                         dtype: Any = None,
                         **kwargs) -> Generator:
  '''
  Factory method for creating a batch generator.
  '''
  if dtype is None:
    dtype = default_float()

  def batch_provider():
    for _ in range(num_batches):
      yield subroutine(batch_shape, dtype=dtype, **kwargs)

  return batch_provider
