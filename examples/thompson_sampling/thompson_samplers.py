#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf
import gpflow

from typing import *
from functools import partial
from .utils import *

# ---- Exports
__all__ = ('ThompsonSampler', 'PathwiseThompsonSampler')


# ==============================================
#                              thompson_samplers
# ==============================================
class ThompsonSampler(tf.Module):
  def __init__(self,
               sampler: Callable,
               input_dim: int,
               num_candidates: int = 2048,
               batch_size: int = 1024,
               num_batches: int = 100):

    self.sampler = sampler
    self.input_dim = input_dim
    self.num_candidates = num_candidates
    self.batch_size = batch_size
    self.num_batches = num_batches

  def __call__(self,
               num_choices: int,
               num_candidates: int = None,
               **kwargs) -> tf.Tensor:

    if num_candidates is None:
      num_candidates = self.num_candidates

    xvals, _ = self.choose_candidates(num_candidates, **kwargs)
    assert list(xvals.shape) == [num_choices, num_candidates, self.input_dim]

    samples = tf.squeeze(self.sampler(xvals, sample_shape=[]), -1)
    choices = take_top_k(arr=xvals, vals=samples, k=1, sign='-')[0]
    return tf.squeeze(choices, axis=-2)

  def choose_candidates(self,
                        num_candidates: int = None,
                        batch_provider: Generator = None,
                        full_cov: bool = False,
                        compile: bool = True,
                        **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

    if num_candidates is None:
      num_candidates = self.num_candidates

    if batch_provider is None:
      batch_shape = self.batch_size, self.input_dim
      batch_provider = build_batch_provider(num_batches=self.num_batches,
                                            batch_shape=batch_shape)

    def _fn(xvals, full_cov=full_cov, **kwargs):
      fvals = self.sampler(xvals, full_cov=full_cov, **kwargs)
      return tf.squeeze(fvals, axis=-1)

    fn = partial(_fn, **kwargs)
    if compile:
      fn = tf.function(fn)

    top_xvals, top_fvals = find_top_k(fn=fn,
                                      batch_provider=batch_provider,
                                      k=num_candidates,
                                      sign='-')

    return top_xvals, tf.expand_dims(top_fvals, axis=-1)


class PathwiseThompsonSampler(ThompsonSampler):
  def __init__(self,
               sampler: Callable,
               input_dim: int,
               num_candidates: int = 32,
               optimizer: Any = None,
               batch_size: int = 256,
               num_batches: int = 100):

    if optimizer is None:
      optimizer = gpflow.optimizers.Scipy()

    self.sampler = sampler
    self.input_dim = input_dim
    self.optimizer = optimizer
    self.num_candidates = num_candidates

    self.batch_size = batch_size
    self.num_batches = num_batches

  def __call__(self,
               num_choices: int,
               num_candidates: int = None,
               optimizer: Any = None,
               compile: bool = True,
               **kwargs) -> tf.Tensor:

    if optimizer is None:
      optimizer = self.optimizer

    xinit = self.choose_candidates(num_candidates, compile=compile, **kwargs)[0]
    xvars = tf.Variable(xinit, constraint=lambda x: tf.clip_by_value(x, 0, 1))

    def _closure():
      fvals = self.sampler(xvars)
      return tf.reduce_sum(fvals)

    closure = tf.function(_closure) if compile else _closure
    bounds = np.full([tf.size(xvars), 2], (0, 1))
    result = optimizer.minimize(closure=closure,
                                variables=[xvars],
                                bounds=bounds)

    fvals = self.sampler(xvars)
    argmins = tf.expand_dims(tf.argmin(fvals, axis=1), 1)
    choices = tf.squeeze(take_along_axis(xvars, argmins, axis=1), 1)
    return choices

  def choose_candidates(self,
                        num_candidates: int = None,
                        batch_provider: Generator = None,
                        compile: bool = True,
                        **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

    if num_candidates is None:
      num_candidates = self.num_candidates

    if batch_provider is None:
      batch_shape = self.batch_size, self.input_dim
      batch_provider = build_batch_provider(num_batches=self.num_batches,
                                            batch_shape=batch_shape)

    def _fn(xvals, **kwargs):
      fvals = self.sampler(xvals, **kwargs)
      return tf.squeeze(fvals, axis=-1)

    fn = partial(_fn, **kwargs)
    if compile:
      fn = tf.function(fn)

    top_xvals, top_fvals = find_top_k(fn=fn,
                                      batch_provider=batch_provider,
                                      k=num_candidates,
                                      sign='-')

    return top_xvals, top_fvals
