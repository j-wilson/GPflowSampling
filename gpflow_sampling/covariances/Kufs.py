#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow.base import TensorLike
from gpflow.covariances.dispatch import Kuf
from gpflow_sampling.utils import swap_axes
from gpflow_sampling.kernels import Conv2d
from gpflow_sampling.covariances.Kfus import Kfu as Kfu_dispatch
from gpflow_sampling.inducing_variables import InducingImages


# ==============================================
#                                           Kufs
# ==============================================
@Kuf.register(InducingImages, Conv2d, TensorLike)
def _Kuf_conv2d_fallback(Z, kernel, X, full_spatial: bool = False, **kwargs):
  Kfu = Kfu_dispatch(Z, kernel, X, full_spatial=full_spatial, **kwargs)

  ndims_x = X.shape.ndims - 3  # assume x lives in 3d image space
  ndims_z = Z.as_images.shape.ndims - 3

  if full_spatial:
    assert Kfu.shape.ndims == ndims_x + ndims_z + 2
    return swap_axes(Kfu, -4, -1)  # TODO: this is a hack

  # Swap the batch axes of x and z
  assert Kfu.shape.ndims == ndims_x + ndims_z
  axes = list(range(ndims_x + ndims_z))
  perm = axes[ndims_x: ndims_x + ndims_z] + axes[:ndims_x]
  return tf.transpose(Kfu, perm)
