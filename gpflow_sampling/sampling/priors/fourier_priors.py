#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import Any, List, Callable
from gpflow.config import default_float
from gpflow.kernels import Kernel, MultioutputKernel
from gpflow.utilities import Dispatcher
from gpflow_sampling.bases import fourier as fourier_basis
from gpflow_sampling.sampling.core import DenseSampler, MultioutputDenseSampler
from gpflow_sampling.kernels import Conv2d, DepthwiseConv2d


# ---- Exports
__all__ = ('random_fourier',)
random_fourier = Dispatcher("random_fourier")


# ==============================================
#                                 fourier_priors
# ==============================================
@random_fourier.register(Kernel)
def _random_fourier(kernel: Kernel,
                    sample_shape: List,
                    num_bases: int,
                    basis: Callable = None,
                    dtype: Any = None,
                    name: str = None,
                    **kwargs):

  if dtype is None:
    dtype = default_float()

  if basis is None:
    basis = fourier_basis(kernel, num_bases=num_bases)

  weights = tf.random.normal(list(sample_shape) + [1, num_bases], dtype=dtype)
  return DenseSampler(weights=weights, basis=basis, name=name, **kwargs)


@random_fourier.register(MultioutputKernel)
def _random_fourier_multioutput(kernel: MultioutputKernel,
                                sample_shape: List,
                                num_bases: int,
                                basis: Callable = None,
                                dtype: Any = None,
                                name: str = None,
                                multioutput_axis: int = 0,
                                **kwargs):
  if dtype is None:
    dtype = default_float()

  if basis is None:
    basis = fourier_basis(kernel, num_bases=num_bases)

  shape = list(sample_shape) + [kernel.num_latent_gps, num_bases]
  weights = tf.random.normal(shape, dtype=dtype)
  return MultioutputDenseSampler(name=name,
                                 basis=basis,
                                 weights=weights,
                                 multioutput_axis=multioutput_axis,
                                 **kwargs)


@random_fourier.register(Conv2d)
def _random_fourier_conv(kernel: Conv2d,
                         sample_shape: List,
                         num_bases: int,
                         basis: Callable = None,
                         dtype: Any = None,
                         name: str = None,
                         **kwargs):

  if dtype is None:
    dtype = default_float()

  if basis is None:
    basis = fourier_basis(kernel, num_bases=num_bases)

  shape = list(sample_shape) + [kernel.num_latent_gps, num_bases]
  weights = tf.random.normal(shape, dtype=dtype)
  return MultioutputDenseSampler(weights=weights,
                                 basis=basis,
                                 name=name,
                                 **kwargs)


@random_fourier.register(DepthwiseConv2d)
def _random_fourier_depthwise_conv(kernel: DepthwiseConv2d,
                                   sample_shape: List,
                                   num_bases: int,
                                   basis: Callable = None,
                                   dtype: Any = None,
                                   name: str = None,
                                   **kwargs):

  if dtype is None:
    dtype = default_float()

  if basis is None:
    basis = fourier_basis(kernel, num_bases=num_bases)

  channels_out = num_bases * kernel.channels_in
  shape = list(sample_shape) + [kernel.num_latent_gps, channels_out]
  weights = tf.random.normal(shape, dtype=dtype)
  return MultioutputDenseSampler(weights=weights,
                                 basis=basis,
                                 name=name,
                                 **kwargs)
