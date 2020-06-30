#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import nlopt
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from typing import *
from sobol_seq import i4_sobol_generate


# ---- Exports
__all__ = (
  'RandomSearch',
  'DIRECT',
  'ThompsonSamplingBandit',
)


# ==============================================
#                                     optimizers
# ==============================================
class RandomSearch:
  def __init__(self,
               task: Callable,
               eval_limit: int,
               data: List[Tuple[tf.Tensor, tf.Tensor]] = None,
               **kwargs):

    if data is None:
      data = list()

    self.task = task
    self.data = data
    self.eval_limit = eval_limit

  def run(self, **kwargs) -> None:
    n = self.eval_limit - len(self.data)
    X = i4_sobol_generate(self.task.input_dim, n, skip=len(self.data))
    Y = self.task(X)
    self.data.extend(zip(X, Y))


class DIRECT:
  def __init__(self,
               task: Callable,
               eval_limit: int,
               data: List[Tuple[tf.Tensor, tf.Tensor]] = None,
               **kwargs):

    if data is None:
      data = list()

    self.task = task
    self.data = data
    self.eval_limit = eval_limit

  def run(self, **kwargs) -> None:
    def closure(arr, *args):
      x = tf.convert_to_tensor(arr.copy(order='C'))
      y = tf.squeeze(self.task(x[None]), axis=0)
      self.data.append((x, y))
      return float(y)

    n = self.eval_limit - len(self.data)
    DIRECT = nlopt.opt(nlopt.GN_DIRECT, self.task.input_dim)
    DIRECT.set_min_objective(closure)
    DIRECT.set_maxeval(n)
    DIRECT.set_lower_bounds(0.0)
    DIRECT.set_upper_bounds(1.0)
    _ = DIRECT.optimize([0.5] * self.task.input_dim)


class ThompsonSamplingBandit:
  def __init__(self,
               task: Callable,
               agent_factory: Callable,
               eval_limit: int,
               par_limit: int = 1,
               data: List[Tuple[tf.Tensor, tf.Tensor]] = None,
               callbacks: List[Callable] = None):

    if data is None:
      data = list()

    if callbacks is None:
      callbacks = list()

    self.task = task
    self.data = data
    self.agent_factory = agent_factory
    self.callbacks = callbacks
    self.par_limit = par_limit
    self.eval_limit = eval_limit

  def run(self, show_progress=False, **kwargs) -> None:
    num_evals = self.eval_limit - len(self.data)
    num_steps = int(np.ceil(num_evals/self.par_limit))
    step_iterator = range(num_steps)
    if show_progress:
      step_iterator = tqdm(step_iterator)

    for _ in step_iterator:
      num_choices = min(self.par_limit, self.eval_limit - len(self.data))
      agent = self.agent_factory(data=self.data, num_choices=num_choices)
      choices = agent(num_choices=num_choices, **kwargs)
      outcomes = self.task(choices)
      self.data.extend(zip(choices, outcomes))
      for func in self.callbacks:
        func(data=self.data)
