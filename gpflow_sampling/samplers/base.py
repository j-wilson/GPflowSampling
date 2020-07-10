#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from abc import abstractmethod
from typing import List, Callable, Union

# ---- Exports
__all__ = (
  'Sampler',
  'CompositeSampler',
  'BayesianLinearSampler',
)


# ==============================================
#                                           base
# ==============================================
class Sampler(tf.Module):
  @abstractmethod
  def __call__(self, *args, **kwargs) -> tf.Tensor:
    raise NotImplementedError

  @abstractmethod
  def reset_random_variables(self, *args, **kwargs):
    raise NotImplementedError


class CompositeSampler(tf.Module):
  def __init__(self,
               join_rule: Callable,
               samplers: List[Sampler],
               mean_function: Callable = None,
               name: str = None) -> 'CompositeSampler':
    super().__init__(name=name)
    self._join_rule = join_rule
    self._samplers = samplers
    self.mean_function = mean_function

  def __call__(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    samples = [sampler(inputs, **kwargs) for sampler in self.samplers]
    outputs = self.join_rule(samples)
    if self.mean_function is None:
      return outputs
    return outputs + self.mean_function(inputs)

  def reset_random_variables(self, *args, **kwargs):
    for sampler in self.samplers:
      sampler.reset_random_variables(*args, **kwargs)

  @property
  def join_rule(self) -> Callable:
    return self._join_rule

  @property
  def samplers(self) -> List[Sampler]:
      return self._samplers


class BayesianLinearSampler(Sampler):
  def __init__(self,
               weights: Union[tf.Tensor, tf.Variable],
               basis: Callable = None,
               mean_function: Callable = None,
               weight_initializer: Callable = None,
               name: str = None):
    """
    Base class for representing samples as a weighted sum of basis
    functions, i.e. $f(x) = basis(x) @ w$.
    """
    super().__init__(name=name)
    self.weights = weights
    self.basis = basis
    self.mean_function = mean_function
    self.weight_initializer = weight_initializer

  def __call__(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    features = inputs if self.basis is None else self.basis(inputs, **kwargs)
    outputs = tf.matmul(features, self.weights, transpose_b=True)
    if self.mean_function is None:
      return outputs
    return outputs + self.mean_function(inputs)

  def reset_random_variables(self, *args, reset_basis: bool = True, **kwargs):
    assert callable(self.weight_initializer)  # [!] improve me
    if reset_basis:
      self.basis.reset_random_variables(*args, **kwargs)

    new_weights = self.weight_initializer(shape=self.weights.shape,
                                          dtype=self.weights.dtype)

    if isinstance(self.weights, tf.Variable):
      self.weights.assign(new_weights)
    else:
      self.weights = new_weights
