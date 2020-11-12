#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
from typing import Union
from gpflow import kernels as gpflow_kernels
from gpflow.base import TensorType
from gpflow.utilities import Dispatcher
from gpflow.inducing_variables import InducingVariables
from gpflow_sampling import kernels
from gpflow_sampling.bases import fourier_bases
from gpflow_sampling.bases.core import KernelBasis


# ---- Exports
__all__ = (
  'kernel_basis',
  'fourier_basis',
)

kernel_basis = Dispatcher("kernel_basis")
fourier_basis = Dispatcher("fourier_basis")


# ==============================================
#                                       dispatch
# ==============================================
@kernel_basis.register(gpflow_kernels.Kernel)
def _kernel_fallback(kern: gpflow_kernels.Kernel,
                     centers: Union[TensorType, InducingVariables],
                     **kwargs):
  return KernelBasis(kernel=kern, centers=centers, **kwargs)


@fourier_basis.register(gpflow_kernels.Stationary)
def _fourier_stationary(kern: gpflow_kernels.Stationary, **kwargs):
  return fourier_bases.Dense(kernel=kern, **kwargs)


@fourier_basis.register(gpflow_kernels.MultioutputKernel)
def _fourier_multioutput(kern: gpflow_kernels.MultioutputKernel, **kwargs):
  return fourier_bases.MultioutputDense(kernel=kern, **kwargs)


@fourier_basis.register(kernels.Conv2d)
def _fourier_conv2d(kern: kernels.Conv2d, **kwargs):
  return fourier_bases.Conv2d(kernel=kern, **kwargs)


@fourier_basis.register(kernels.Conv2dTranspose)
def _fourier_conv2d_transposed(kern: kernels.Conv2dTranspose, **kwargs):
  return fourier_bases.Conv2dTranspose(kernel=kern, **kwargs)


@fourier_basis.register(kernels.DepthwiseConv2d)
def _fourier_depthwise_conv2d(kern: kernels.DepthwiseConv2d, **kwargs):
  return fourier_bases.DepthwiseConv2d(kernel=kern, **kwargs)
