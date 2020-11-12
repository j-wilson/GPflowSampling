#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from gpflow import kernels
from gpflow.base import TensorLike
from gpflow.utilities import Dispatcher
from gpflow.inducing_variables import (InducingVariables,
                                       SharedIndependentInducingVariables)
from gpflow.covariances.dispatch import Kuf as Kuf_dispatch
from gpflow_sampling.kernels import Conv2d, DepthwiseConv2d
from gpflow_sampling.utils import move_axis, get_inducing_shape
from gpflow_sampling.inducing_variables import (InducingImages,
                                                DepthwiseInducingImages)


# ==============================================
#                                           Kfus
# ==============================================
Kfu = Dispatcher("Kfu")


@Kfu.register(InducingVariables, kernels.Kernel, TensorLike)
def _Kfu_fallback(Z, kern, X, **kwargs):
  Kuf = Kuf_dispatch(Z, kern, X, **kwargs)

  # Assume features of x and z are 1-dimensional
  ndims_x = X.shape.ndims - 1  # assume x lives in 1d space
  ndims_z = len(get_inducing_shape(Z)) - 1
  assert ndims_x + ndims_z == Kuf.shape.ndims

  # Swap the batch axes of x and z
  axes = list(range(ndims_x + ndims_z))
  perm = axes[ndims_z: ndims_z + ndims_x] + axes[:ndims_z]
  return tf.transpose(Kuf, perm)


@Kfu.register(InducingVariables, kernels.MultioutputKernel, TensorLike)
def _Kfu_fallback_multioutput(Z, kern, X, **kwargs):
  Kuf = Kuf_dispatch(Z, kern, X, **kwargs)

  # Assume features of x and z are 1-dimensional
  ndims_x = X.shape.ndims - 1  # assume x lives in 1d space
  ndims_z = 1  # shared Z live in 1d space, separate Z are 2d but 1-to-1 with L
  assert ndims_x + ndims_z == Kuf.shape.ndims - 1

  # Swap the batch axes of x and z
  axes = list(range(1, ndims_x + ndims_z + 1))  # keep L output-features first
  perm = [0] + axes[ndims_z: ndims_z + ndims_x] + axes[:ndims_z]
  return tf.transpose(Kuf, perm)


@Kfu.register(SharedIndependentInducingVariables,
              kernels.SharedIndependent,
              TensorLike)
def _Kfu_fallback_shared(Z, kern, X, **kwargs):
  return _Kfu_fallback(Z, kern, X, **kwargs)  # Edge-case where L is supressed


def _Kfu_conv2d_fallback(feat: InducingImages,
                         kern: Conv2d,
                         Xnew: tf.Tensor,
                         full_spatial: bool = False):

  Zp = feat.as_patches  # [M, patch_len]
  Xp = kern.get_patches(Xnew, full_spatial=full_spatial)
  Kxz = kern.kernel.K(Xp, Zp)  # [N, H * W, M] or [N, H, W, M]
  if full_spatial:  # convert to 4D image format
    spatial_out = kern.get_spatial_out(Xnew.shape[-3:-1])  # to [N, H, W, M]
    return tf.reshape(Kxz, list(Kxz.shape[:-2]) + spatial_out + [Kxz.shape[-1]])

  if kern.weights is None:
    return tf.reduce_mean(Kxz, axis=-2)

  return tf.tensordot(Kxz, kern.weights, [-2, -1])


@Kfu.register(InducingImages, Conv2d, object)
def _Kfu_conv2d(feat: InducingImages,
                kern: Conv2d,
                Xnew: tf.Tensor,
                full_spatial: bool = False):

  if not isinstance(kern.kernel, kernels.Stationary):
    return _Kfu_conv2d_fallback(feat, kern, Xnew, full_spatial)

  # Compute (squared) Mahalanobis distances between patches
  patch_shape = list(kern.patch_shape)
  channels_in = Xnew.shape[-3 if kern.data_format == "NCHW" else -1]
  precis = tf.square(tf.math.reciprocal(kern.kernel.lengthscales))

  # Construct lengthscale filters [h, w, channels_in, 1]
  if kern.kernel.ard:
    filters = tf.reshape(precis, patch_shape + [channels_in, 1])
  else:
    filters = tf.fill(patch_shape + [channels_in, 1], precis)

  r2 = tf.transpose(tf.nn.conv2d(input=tf.square(feat.as_images),
                                 filters=filters,
                                 strides=[1, 1],
                                 padding="VALID"))

  X = tf.reshape(Xnew, [-1] + list(Xnew.shape)[-3:])  # stack as 4d images
  r2 += kern.convolve(tf.square(X), filters)  # [N, height_out, width_out, M]

  filters *= feat.as_filters  # [h, w, channels_in, M]
  r2 -= 2 * kern.convolve(X, filters)

  Kxz = kern.kernel.K_r2(r2)
  if not full_spatial:
    Kxz = tf.reshape(Kxz, list(Kxz.shape[:-3]) + [-1, len(feat)])  # [N, P, M]
    if kern.weights is None:
      Kxz = tf.reduce_mean(Kxz, axis=-2)
    else:
      Kxz = tf.tensordot(Kxz, kern.weights, [-2, -1])

  # Undo stacking of Xnew as 4d images X
  return tf.reshape(Kxz, list(Xnew.shape[:-3]) + list(Kxz.shape[1:]))


def _Kfu_depthwise_conv2d_fallback(feat: DepthwiseInducingImages,
                                   kern: DepthwiseConv2d,
                                   Xnew: tf.Tensor,
                                   full_spatial: bool = False):

  Zp = feat.as_patches  # [M, channels_in, patch_len]
  Xp = kern.get_patches(Xnew, full_spatial=full_spatial)
  r2 = tf.reduce_sum(tf.math.squared_difference(  # compute square distances
          tf.expand_dims(kern.kernel.scale(Xp), -Zp.shape.ndims),
          kern.kernel.scale(Zp)), axis=-1)

  Kxz = kern.kernel.K_r2(r2)
  if full_spatial:  # convert to 4D image format as [N, H, W, channels_in * M]
    return tf.reshape(move_axis(Kxz, -1, -2), list(Kxz.shape[:-2]) + [-1])

  if kern.weights is None:  # reduce over channels and patches
    return tf.reduce_mean(Kxz, axis=[-3, -1])

  return tf.tensordot(kern.weights, Kxz, axes=[(0, 1), (-3, -1)])


@Kfu.register(DepthwiseInducingImages, DepthwiseConv2d, object)
def _Kfu_depthwise_conv2d(feat: DepthwiseInducingImages,
                          kern: DepthwiseConv2d,
                          Xnew: tf.Tensor,
                          full_spatial: bool = False):

  if not isinstance(kern.kernel, kernels.Stationary):
    return _Kfu_depthwise_conv2d_fallback(feat, kern, Xnew, full_spatial)

  # Compute (squared) Mahalanobis distances between patches
  patch_shape = list(kern.patch_shape)
  channels_in = Xnew.shape[-3 if kern.data_format == "NCHW" else -1]
  channels_out = len(feat) * channels_in
  precis = tf.square(tf.math.reciprocal(kern.kernel.lengthscales))

  # Construct lengthscale filters [h, w, channels_in, 1]
  if kern.kernel.ard:  # notice the transpose!
    assert tuple(precis.shape) == (channels_in, tf.reduce_prod(patch_shape))
    filters = tf.reshape(tf.transpose(precis), patch_shape + [channels_in, 1])
  else:
    filters = tf.fill(patch_shape + [channels_in, 1], precis)

  ZZ = tf.nn.depthwise_conv2d(input=tf.square(feat.as_images),
                              filter=filters,
                              strides=[1, 1, 1, 1],
                              padding="VALID")  # [M, 1, 1, channels_in]

  r2 = tf.reshape(move_axis(ZZ, 0, -1), [1, 1, 1, channels_out])

  X = tf.reshape(Xnew, [-1] + list(Xnew.shape)[-3:])  # stack as 4d images
  r2 += tf.repeat(kern.convolve(tf.square(X), filters), len(feat), axis=-1)

  filters *= feat.as_filters  # [h, w, channels_in, M]
  r2 -= 2 * kern.convolve(X, filters)  # [N, height_out, width_out, chan_out]

  Kxz = kern.kernel.K_r2(r2)
  if full_spatial:
    Kxz = tf.reduce_mean(
                tf.reshape(Kxz, list(Kxz.shape[:-1]) + [channels_in, -1]),
                axis=-2)  # average over input channels
  else:
    Kxz = tf.reshape(Kxz, list(Kxz.shape[:-3]) + [-1, len(feat)])  # [N, P, M]
    if kern.weights is None:
      Kxz = tf.reduce_mean(Kxz, axis=-2)
    else:
      div = tf.cast(1/channels_in, Kxz.dtype)
      Kxz = div * tf.tensordot(Kxz, tf.reshape(kern.weights, [-1]), [-2, -1])

  # Undo stacking of Xnew as 4d images X
  return tf.reshape(Kxz, list(Xnew.shape[:-3]) + list(Kxz.shape[1:]))
