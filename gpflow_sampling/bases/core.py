#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
from abc import abstractmethod
from typing import Union
from tensorflow import Module
from gpflow.base import TensorType
from gpflow.kernels import Kernel, SharedIndependent, SeparateIndependent
from gpflow.inducing_variables import InducingVariables
from gpflow_sampling.utils import get_inducing_shape
from gpflow_sampling.covariances import Kfu as Kfu_dispatch

# ---- Exports
__all__ = ('AbstractBasis', 'KernelBasis')


# ==============================================
#                                           core
# ==============================================
class AbstractBasis(Module):
  def __init__(self, initialized: bool = False, name: str = None):
    super().__init__(name=name)
    self.initialized = initialized

  @abstractmethod
  def __call__(self, *args, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def initialize(self, *args, **kwargs):
    pass

  def _maybe_initialize(self, *args, **kwargs):
    if not self.initialized:
      self.initialize(*args, **kwargs)
      self.initialized = True

  @property
  @abstractmethod
  def num_bases(self):
    raise NotImplementedError


class KernelBasis(AbstractBasis):
  def __init__(self,
               kernel: Kernel,
               centers: Union[TensorType, InducingVariables],
               name: str = None,
               **default_kwargs):

    super().__init__(name=name)
    self.kernel = kernel
    self.centers = centers
    self.default_kwargs = default_kwargs

  def __call__(self, x, **kwargs):
    _kwargs = {**self.default_kwargs, **kwargs}  # resolve keyword arguments
    self._maybe_initialize(x, **_kwargs)
    if isinstance(self.centers, InducingVariables):
      return Kfu_dispatch(self.centers, self.kernel, x, **_kwargs)

    if isinstance(self.kernel, (SharedIndependent, SeparateIndependent)):
      # TODO: Improve handling of "full_output_cov". Here, we're imitating
      #       the behavior of gpflow.covariances.Kuf.
      _kwargs.setdefault('full_output_cov', False)

    return self.kernel.K(x, self.centers, **_kwargs)

  @property
  def num_bases(self):
    """
    TODO: Edge-cases?
    """
    if isinstance(self.centers, InducingVariables):
      return get_inducing_shape(self.centers)[-1]
    return self.centers.shape[-1]
