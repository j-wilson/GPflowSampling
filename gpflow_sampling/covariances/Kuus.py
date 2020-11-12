#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow.utilities.ops import square_distance
from gpflow.covariances.dispatch import Kuu
from gpflow_sampling.utils import move_axis
from gpflow_sampling.kernels import Conv2d, DepthwiseConv2d
from gpflow_sampling.inducing_variables import (InducingImages,
                                                DepthwiseInducingImages)

# ==============================================
#                                           Kuus
# ==============================================
@Kuu.register(InducingImages, Conv2d)
def _Kuu_conv2d(feat: InducingImages,
                kern: Conv2d,
                jitter: float = 0.0):
  _Kuu = kern.kernel.K(feat.as_patches)
  return tf.linalg.set_diag(_Kuu, tf.linalg.diag_part(_Kuu) + jitter)


@Kuu.register(DepthwiseInducingImages, DepthwiseConv2d)
def _Kuu_depthwise_conv2d(feat: DepthwiseInducingImages,
                          kern: DepthwiseConv2d,
                          jitter: float = 0.0):

  # Prepare scaled inducing patches; shape(Zp) = [channels_in, M, patch_len]
  Zp = move_axis(kern.kernel.scale(feat.as_patches), -2, 0)
  r2 = square_distance(Zp, None)
  _Kuu = tf.reduce_mean(kern.kernel.K_r2(r2), axis=0)  # [M, M]
  return tf.linalg.set_diag(_Kuu, tf.linalg.diag_part(_Kuu) + jitter)
