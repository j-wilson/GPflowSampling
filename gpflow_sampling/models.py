#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from abc import abstractmethod
from typing import Optional
from contextlib import contextmanager
from gpflow.base import TensorLike
from gpflow.config import default_float, default_jitter
from gpflow.models import GPModel, SVGP, GPR
from gpflow_sampling import sampling, covariances
from gpflow_sampling.sampling.core import AbstractSampler, CompositeSampler


# ---- Exports
__all__ = ('PathwiseGPModel', 'PathwiseGPR', 'PathwiseSVGP')


# ==============================================
#                                         models
# ==============================================
class PathwiseGPModel(GPModel):
  def __init__(self, *args, paths: AbstractSampler = None, **kwargs):
    super().__init__(*args, **kwargs)
    self._paths = paths

  @abstractmethod
  def generate_paths(self, *args, **kwargs) -> AbstractSampler:
    raise NotImplementedError

  def predict_f_samples(self,
                        Xnew: TensorLike,
                        num_samples: Optional[int] = None,
                        full_cov: bool = True,
                        full_output_cov: bool = True,
                        **kwargs) -> tf.Tensor:

    assert full_cov and full_output_cov, NotImplementedError
    if self.paths is None:
      raise RuntimeError("Paths were not initialized.")

    if num_samples is not None:
      assert num_samples == self.paths.sample_shape,\
        ValueError("Requested number of samples does not match path count.")

    return self.paths(Xnew, **kwargs)

  @contextmanager
  def temporary_paths(self, *args, **kwargs):
    try:
      init_paths = self.paths
      temp_paths = self.generate_paths(*args, **kwargs)
      self.set_paths(temp_paths)
      yield temp_paths
    finally:
      self.set_paths(init_paths)

  def set_paths(self, paths) -> AbstractSampler:
    self._paths = paths
    return paths

  @contextmanager
  def set_temporary_paths(self, paths):
    try:
      init_paths = self.paths
      self.set_paths(paths)
      yield paths
    finally:
      self.set_paths(init_paths)

  @property
  def paths(self) -> AbstractSampler:
    return self._paths


class PathwiseGPR(GPR, PathwiseGPModel):
  def __init__(self, *args, paths: AbstractSampler = None, **kwargs):
    GPR.__init__(self, *args, **kwargs)
    self._paths = paths

  def generate_paths(self,
                     num_samples: int,
                     num_bases: int = None,
                     prior: AbstractSampler = None,
                     sample_axis: int = None,
                     **kwargs) -> CompositeSampler:

    if prior is None:
      prior = sampling.priors.random_fourier(self.kernel,
                                             num_bases=num_bases,
                                             sample_shape=[num_samples],
                                             sample_axis=sample_axis)
    elif num_bases is not None:
      assert prior.sample_shape == [num_samples]

    diag = tf.convert_to_tensor(self.likelihood.variance)
    return sampling.decoupled(self.kernel,
                              prior,
                              *self.data,
                              mean_function=self.mean_function,
                              diag=diag,
                              sample_axis=sample_axis,
                              **kwargs)


class PathwiseSVGP(SVGP, PathwiseGPModel):
  def __init__(self, *args, paths: AbstractSampler = None, **kwargs):
    SVGP.__init__(self, *args, **kwargs)
    self._paths = paths

  def generate_paths(self,
                     num_samples: int,
                     num_bases: int = None,
                     prior: AbstractSampler = None,
                     sample_axis: int = None,
                     **kwargs) -> CompositeSampler:

    if prior is None:
      prior = sampling.priors.random_fourier(self.kernel,
                                             num_bases=num_bases,
                                             sample_shape=[num_samples],
                                             sample_axis=sample_axis)
    elif num_bases is not None:
      assert prior.sample_shape == [num_samples]

    return sampling.decoupled(self.kernel,
                              prior,
                              self.inducing_variable,
                              self._generate_u(num_samples),
                              mean_function=self.mean_function,
                              sample_axis=sample_axis,
                              **kwargs)

  def _generate_u(self, num_samples: int, L: tf.Tensor = None):
    """
    Returns samples $u ~ q(u)$.
    """
    q_sqrt = tf.linalg.band_part(self.q_sqrt, -1, 0)
    shape = self.num_latent_gps, q_sqrt.shape[-1], num_samples
    rvs = tf.random.normal(shape, dtype=default_float())  # [L, M, S]
    uT = q_sqrt @ rvs + tf.transpose(self.q_mu)[..., None]
    if self.whiten:
      if L is None:
        Z = self.inducing_variable
        K = covariances.Kuu(Z, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(K)
      uT = L @ uT
    return tf.transpose(uT)  # [S, M, L]
