#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import numpy as np
import tensorflow as tf
import torch
import gpflow
import gpytorch

from gpflow.utilities import Dispatcher
from typing import *

# ---- Exports
convert_kernel = Dispatcher("convert_kernel")
convert_likelihood = Dispatcher("convert_likelihood")
__all__ = (
  'convert_kernel',
  'convert_likelihood',
  'GridGPR',
  'GPyTorchSampler',
)

# ==============================================
#                                gpytorch_extras
# ==============================================
@convert_kernel.register(gpflow.kernels.SquaredExponential)
def convert_kernel_squaredExp(kernel, **kwargs):
  base_kernel = gpytorch.kernels.RBFKernel(**kwargs)
  base_kernel.lengthscale = kernel.lengthscales.numpy()

  scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel=base_kernel)
  scaled_kernel.outputscale = kernel.variance.numpy()
  return scaled_kernel

@convert_kernel.register(gpflow.kernels.Matern52)
def convert_kernel_matern52(kernel, nu=None, **kwargs):
  assert nu is None
  base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, **kwargs)
  base_kernel.lengthscale = kernel.lengthscales.numpy()

  scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel=base_kernel)
  scaled_kernel.outputscale = kernel.variance.numpy()
  return scaled_kernel


@convert_kernel.register(gpflow.kernels.Matern32)
def convert_kernel_matern52(kernel, nu=None, **kwargs):
  assert nu is None
  base_kernel = gpytorch.kernels.MaternKernel(nu=1.5, **kwargs)
  base_kernel.lengthscale = kernel.lengthscales.numpy()

  scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel=base_kernel)
  scaled_kernel.outputscale = kernel.variance.numpy()
  return scaled_kernel


@convert_kernel.register(gpflow.kernels.Matern12)
def convert_kernel_matern52(kernel, nu=None, **kwargs):
  assert nu is None
  base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, **kwargs)
  base_kernel.lengthscale = kernel.lengthscales.numpy()

  scaled_kernel = gpytorch.kernels.ScaleKernel(base_kernel=base_kernel)
  scaled_kernel.outputscale = kernel.variance.numpy()
  return scaled_kernel


@convert_likelihood.register(gpflow.likelihoods.Gaussian)
def convert_likelihood_gaussian(likelihood, **kwargs):
  _likelihood = gpytorch.likelihoods.GaussianLikelihood(**kwargs)
  _likelihood.noise = likelihood.variance.numpy()
  return _likelihood


class GridGPR(gpflow.models.GPR):
  def __init__(self,
               likelihood: gpflow.likelihoods.Likelihood,
               kernel: gpflow.kernels.Kernel,
               input_dim: int,
               grid_size: int):
    """
    Creates exact GP regression models in both GPflow and GPyTorch,
    whose training data $y = N(f(X), \sigma^{2} I)$ is drawn from
    the given GP prior, where $X$ is a regularly spaced grid of points.
    """
    _kernel = convert_kernel(kernel)
    _likelihood = convert_likelihood(likelihood)
    self.gpytorch_model = self._create_gpytorch_model(_kernel,
                                                      _likelihood,
                                                      input_dim=input_dim,
                                                      grid_size=grid_size)

    def convert_torchTensor(tensor):
      return np.asarray(tensor.detach(), dtype=gpflow.config.default_float())

    data = convert_torchTensor(self.gpytorch_model.train_inputs[0]), \
           convert_torchTensor(self.gpytorch_model.train_targets)[..., None]

    super().__init__(data=data, kernel=kernel)
    self.likelihood = likelihood

  @classmethod
  def _create_gpytorch_model(cls, kernel, likelihood, input_dim, grid_size):
    grid_bounds = input_dim * [(0, 1)]
    grid_kernel = gpytorch.kernels.GridInterpolationKernel(kernel,
                                                           num_dims=input_dim,
                                                           grid_size=grid_size,
                                                           grid_bounds=grid_bounds)

    # Create training data on the grid
    X = gpytorch.utils.grid.create_data_from_grid(grid_kernel.grid)
    S = grid_kernel._inducing_forward(last_dim_is_batch=False)
    S += likelihood.noise * torch.eye(S.shape[-1])
    y = S.cholesky() @ torch.randn(*S.shape[:-1], 1)[..., 0]

    class GPyTorchGPR(gpytorch.models.ExactGP):
      def __init__(self, X, y, likelihood, kernel):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

      def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    return GPyTorchGPR(X, y, likelihood, grid_kernel)


class GPyTorchSampler:
  def __init__(self,
               model: gpytorch.models.GP,
               sample_shape: List[int],
               max_root_decomposition: int = None):
    """
    Use LanczOs Variance Estimates (LOVE) to generate fast
    samples from an approximate GP posterior.
    """
    self.model = model
    self.sample_shape = sample_shape
    self.max_root_decomposition = max_root_decomposition

  @classmethod
  def build(cls, model: GridGPR, sample_shape, **kwargs):
    return cls(model=model.gpytorch_model,
               sample_shape=sample_shape,
               **kwargs)

  def __call__(self, x):
    self.model.eval()
    if isinstance(x, tf.Tensor):
      x = x.numpy()

    if isinstance(x, np.ndarray):
      x = torch.Tensor(x)

    max_root = self.max_root_decomposition or len(self.model.train_targets)
    with torch.no_grad(), \
         gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.fast_pred_samples(), \
         gpytorch.settings.max_root_decomposition_size(max_root):

      distrib = self.model(x)
      samples = distrib.rsample(torch.Size(self.sample_shape))

    return samples.numpy()[..., None]
