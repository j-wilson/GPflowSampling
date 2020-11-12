#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from abc import abstractmethod
from typing import List, Callable, Union
from gpflow.inducing_variables import InducingVariables
from gpflow_sampling.utils import (move_axis,
                                   normalize_axis,
                                   batch_tensordot,
                                   get_inducing_shape)

# ---- Exports
__all__ = (
  'AbstractSampler',
  'DenseSampler',
  'MultioutputDenseSampler',
  'CompositeSampler',
)


# ==============================================
#                                           core
# ==============================================
class AbstractSampler(tf.Module):
  @abstractmethod
  def __call__(self, *args, **kwargs):
    raise NotImplementedError

  @property
  def sample_shape(self):
    raise NotImplementedError


class CompositeSampler(AbstractSampler):
  def __init__(self,
               join_rule: Callable,
               samplers: List[Callable],
               mean_function: Callable = None,
               name: str = None):
    """
    Combine base samples via a specified join rule.
    """
    super().__init__(name=name)
    self._join_rule = join_rule
    self._samplers = samplers
    self.mean_function = mean_function

  def __call__(self, x: tf.Tensor, **kwargs) -> tf.Tensor:
    samples = [sampler(x, **kwargs) for sampler in self.samplers]
    vals = self.join_rule(samples)
    return vals if self.mean_function is None else vals + self.mean_function(x)

  @property
  def join_rule(self) -> Callable:
    return self._join_rule

  @property
  def samplers(self):
    return self._samplers

  @property
  def sample_shape(self):
    for i, sampler in enumerate(self.samplers):
      if i == 0:
        sample_shape = sampler.sample_shape
      else:
        assert sample_shape == sampler.sample_shape
    return sample_shape


class DenseSampler(AbstractSampler):
  def __init__(self,
               weights: Union[tf.Tensor, tf.Variable],
               basis: Callable = None,
               mean_function: Callable = None,
               sample_axis: int = None,
               name: str = None):
    """
    Return samples as weighted sums of features.
    """
    assert weights.shape.ndims > 1
    super().__init__(name=name)
    self.weights = weights
    self.basis = basis
    self.mean_function = mean_function
    self.sample_axis = sample_axis

  def __call__(self, x: tf.Tensor, sample_axis: int = "default", **kwargs):
    """
    :param sample_axis: Specify an axis of inputs x as corresponding 1-to-1 with
           sample-specific slices of weight tensor w when computing tensor dot
           products.
    """
    if sample_axis == "default":
      sample_axis = self.sample_axis

    feat = x if self.basis is None else self.basis(x, **kwargs)
    if sample_axis is None:
      batch_axes = None
    else:
      assert len(self.sample_shape), "Received sample_axis but self.weights has" \
                                      " no dedicated axis for samples; this" \
                                      " usually implies that sample_shape=[]."

      ndims_x = len(get_inducing_shape(x) if
                    isinstance(x, InducingVariables) else x.shape)

      batch_axes = [-3, normalize_axis(sample_axis, ndims_x) - ndims_x]

    # Batch-axes notwithstanding, shape(vals) = [N, S] (after move_axis)
    vals = move_axis(batch_tensordot(self.weights,
                                     feat,
                                     axes=[-1, -1],
                                     batch_axes=batch_axes),
                     len(self.sample_shape),  # axis=-2 houses scalar output 1
                     -1)

    return vals if self.mean_function is None else vals + self.mean_function(x)

  @property
  def sample_shape(self):
    w_shape = list(self.weights.shape)
    if len(w_shape) == 2:
      return []
    return w_shape[:-2]


class MultioutputDenseSampler(DenseSampler):
  def __init__(self,
               weights: Union[tf.Tensor, tf.Variable],
               *args,
               multioutput_axis: int = None,
               **kwargs):
    """
    :param multioutput_axis: Specify an axis of inputs x (or features thereof)
           as corresponding 1-to-1 with output-feature-specific slices of weight
           tensor w when computing tensor dot products.
    """
    self.multioutput_axis = multioutput_axis
    super().__init__(weights, *args, **kwargs)

  def __call__(self, x: tf.Tensor, sample_axis: int = "default", **kwargs):
    """
    :param sample_axis: Specify an axis of inputs x as corresponding 1-to-1 with
           sample-specific slices of weight tensor w when computing tensor dot
           products.

    TODO: Improve hard-coding of multioutput-/sample-axis of weights.
    """
    if sample_axis == "default":
      sample_axis = self.sample_axis

    if self.multioutput_axis is None:
      batch_w = []  # batch axes for w
      batch_x = []  # batch axes for x
    else:
      batch_w = [-2]
      batch_x = [self.multioutput_axis]

    if sample_axis is not None:
      assert len(self.sample_shape), "Received sample_axis but self.weights has" \
                                      " no dedicated axis for samples; this" \
                                      " usually implies that sample_shape=[]."
      batch_w.append(-3)

      # TODO: If basis(x) grows the rank of x, it should only do so from the
      #       left, such that the negative i-th axis (i > 1) remains the same.
      ndims_x = len(get_inducing_shape(x) if
                    isinstance(x, InducingVariables) else x.shape)
      batch_x.append(normalize_axis(sample_axis, ndims_x) - ndims_x)

    feat = x if self.basis is None else self.basis(x, **kwargs)
    vals = move_axis(batch_tensordot(self.weights,  # output features go last
                                     feat,
                                     axes=[-1, -1],
                                     batch_axes=[batch_w, batch_x]),
                     len(self.sample_shape),  # axis=-2 houses multioutputs L
                     -1)

    return vals if self.mean_function is None else vals + self.mean_function(x)
