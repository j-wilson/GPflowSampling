#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf

from typing import List
from warnings import warn
from gpflow import kernels
from gpflow.base import TensorType, Parameter
from gpflow.config import default_float
from gpflow_sampling.utils import (conv_ops,
                                   swap_axes,
                                   move_axis,
                                   batch_tensordot)
from tensorflow.python.keras.utils.conv_utils import (conv_output_length,
                                                      deconv_output_length)

# ---- Exports
__all__ = (
  'Conv2d',
  'Conv2dTranspose',
  'DepthwiseConv2d',
)


# ==============================================
#                                        kernels
# ==============================================
class Conv2d(kernels.MultioutputKernel):
  def __init__(self,
               kernel: kernels.Kernel,
               image_shape: List,
               patch_shape: List,
               channels_in: int = 1,
               channels_out: int = 1,
               weights: TensorType = "default",
               strides: List = None,
               padding: str = "VALID",
               dilations: List = None,
               data_format: str = "NHWC"):

    strides = list((1, 1) if strides is None else strides)
    dilations = list((1, 1) if dilations is None else dilations)

    # Sanity checks
    assert len(strides) == 2
    assert len(dilations) == 2
    assert padding in ("VALID", "SAME")
    assert data_format in ("NHWC", "NCHW")

    if isinstance(weights, str) and weights == "default":  # TODO: improve me
      spatial_out = self.get_spatial_out(spatial_in=image_shape,
                                         filter_shape=patch_shape,
                                         strides=strides,
                                         padding=padding,
                                         dilations=dilations)

      weights = tf.ones([tf.reduce_prod(spatial_out)], dtype=default_float())

    super().__init__()
    self.kernel = kernel
    self.image_shape = image_shape
    self.patch_shape = patch_shape
    self.channels_in = channels_in
    self.channels_out = channels_out

    self.strides = strides
    self.padding = padding
    self.dilations = dilations
    self.data_format = data_format
    self._weights = None if (weights is None) else Parameter(weights)

  def __call__(self,
               X: TensorType,
               X2: TensorType=None,
               *,
               full_cov: bool = False,
               full_spatial: bool = False,
               presliced: bool = False):

    if not presliced:
      X, X2 = self.slice(X, X2)

    if not full_cov and X2 is not None:
      raise ValueError(
          "Ambiguous inputs: passing in `X2` is not compatible with `full_cov=False`."
      )

    if not full_cov:
      return self.K_diag(X, full_spatial=full_spatial)
    return self.K(X, X2, full_spatial=full_spatial)

  def K(self, X: tf.Tensor, X2: tf.Tensor = None, full_spatial: bool = False):
    """
    TODO: For stationary kernels, implement this using convolutions?
    """
    P = self.get_patches(X, full_spatial=full_spatial)
    P2 = P if X2 is None else self.get_patches(X2, full_spatial=full_spatial)
    K = self.kernel.K(P, P2)
    if full_spatial:
      return K  # [N, H1, W1, N2, H2, W2]

    # At this point, shape(K) = [N, H1 * W1, N2, H2 * W2]
    if self.weights is None:
      return tf.reduce_mean(K, axis=[-3, -1])
    return tf.tensordot(tf.linalg.matvec(K, self.weights),
                        self.weights,
                        axes=[-2, 0])

  def K_diag(self, X: tf.Tensor, full_spatial: bool = False):
    P = self.get_patches(X, full_spatial=full_spatial)
    K = self.kernel.K_diag(P)
    if full_spatial:
      return K  # [N, H1, W1]

    # At this point, shape(K) = [N, H1 * W1]
    if self.weights is None:
      return tf.reduce_mean(K, axis=[-2, -1])
    return tf.linalg.matvec(tf.linalg.matvec(K, self.weights), self.weights)

  def convolve(self,
               input,
               filters,
               strides: List = None,
               padding: str = None,
               dilations: List = None,
               data_format: str = None):

    if strides is None:
      strides = self.strides

    if padding is None:
      padding = self.padding

    if dilations is None:
      dilations = self.dilations

    if data_format is None:
      data_format = self.data_format

    return tf.nn.conv2d(input=input,
                        filters=filters,
                        strides=strides,
                        padding=padding,
                        dilations=dilations,
                        data_format=data_format)

  def get_patches(self, X: TensorType, full_spatial: bool = True):
    # Extract image patches
    X_nchw = conv_ops.reformat_data(X, self.data_format, "NHWC")
    patches = tf.image.extract_patches(images=X_nchw,
                                       sizes=[1] + self.patch_shape + [1],
                                       strides=[1] + self.strides + [1],
                                       rates=[1] + self.dilations + [1],
                                       padding=self.padding)

    if full_spatial:
      output_shape = list(X.shape[:-3]) + list(patches.shape[-3:])
    else:
      output_shape = list(X.shape[:-3]) + [-1, patches.shape[-1]]

    return tf.reshape(patches, output_shape)

  def get_shape_out(self,
                    shape_in: List,
                    filter_shape: List,
                    strides: List = None,
                    dilations: List = None,
                    data_format: str = None) -> List:

    if data_format is None:
      data_format = self.data_format

    if data_format == "NHWC":
      *batch, height, width, _ = list(shape_in)
    else:
      *batch, _, height, width = list(shape_in)

    spatial_out = self.get_spatial_out(spatial_in=[height, width],
                                       filter_shape=filter_shape[:2],
                                       strides=strides,
                                       dilations=dilations)

    nhwc_out = batch + spatial_out + [filter_shape[-1]]
    return conv_ops.reformat_shape(nhwc_out, "NHWC", data_format)

  def get_spatial_out(self,
                      spatial_in: List = None,
                      filter_shape: List = None,
                      strides: List = None,
                      padding: str = None,
                      dilations: List = None) -> List:

    if spatial_in is None:
      spatial_in = self.image_shape

    if filter_shape is None:
      filter_shape = self.patch_shape
    else:
      assert len(filter_shape) == 2

    if strides is None:
      strides = self.strides

    if padding is None:
      padding = self.padding

    if dilations is None:
      dilations = self.dilations

    return [conv_output_length(input_length=spatial_in[i],
                               filter_size=filter_shape[i],
                               stride=strides[i],
                               padding=padding.lower(),
                               dilation=dilations[i]) for i in range(2)]

  @property
  def num_patches(self):
    return tf.reduce_prod(self.get_spatial_out())

  @property
  def weights(self):
    if self._weights is None:
      return None
    return tf.cast(1/self.num_patches, self._weights.dtype) * self._weights

  @property
  def num_latent_gps(self):
    return self.channels_out

  @property
  def latent_kernels(self):
    return self.kernel,


class Conv2dTranspose(Conv2d):
  def __init__(self,
               *args,
               strides: List = None,
               dilations: List = None,
               **kwargs):

    strides = list((1, 1) if strides is None else strides)
    dilations = list((1, 1) if dilations is None else dilations)
    if strides != [1, 1] and dilations != [1, 1]:
        raise NotImplementedError('Tensorflow does not implement transposed'
                                  'convolutions with strides and dilations.')

    super().__init__(*args, strides=strides, dilations=dilations, **kwargs)

  def convolve(self,
               input,
               filters,
               strides: List = None,
               padding: str = None,
               dilations: List = None,
               data_format: str = None):

    if strides is None:
      strides = self.strides

    if padding is None:
      padding = self.padding

    if dilations is None:
      dilations = self.dilations

    if data_format is None:
      data_format = self.data_format

    shape_out = self.get_shape_out(shape_in=input.shape,
                                   filter_shape=filters.shape,
                                   strides=strides,
                                   dilations=dilations,
                                   data_format=data_format)

    _filters = swap_axes(filters, -2, -1)
    conv_kwargs = dict(filters=_filters,
                       padding=padding,
                       output_shape=shape_out)

    if dilations != [1, 1]:  # TODO: improve me
      assert data_format == 'NHWC'
      assert list(strides) == [1, 1]
      assert len(dilations) == 2 and dilations[0] == dilations[1]
      return tf.nn.atrous_conv2d_transpose(input,
                                           rate=dilations[0],
                                           **conv_kwargs)

    return tf.nn.conv2d_transpose(input,
                                  strides=strides,
                                  dilations=dilations,
                                  data_format=data_format,
                                  **conv_kwargs)

  def get_patches(self, X, full_spatial: bool = False):
    """
    Returns the patches used by a 2d transposed convolution.
    """
    spatial_in = X.shape[-3: -1]

    # Pad X with (stride - 1) zeros in between each pixel
    if any(stride != 1 for stride in self.strides):
      shape = list(X.shape[:-3])
      terms = [tf.range(size) for size in shape]
      for i, stride in enumerate(self.strides):
        size = X.shape[i - 3]
        shape.append(stride * (size - 1) + 1)
        terms.append(tf.range(stride * size, delta=stride))
      shape.append(X.shape[-1])
      grid = tf.meshgrid(*terms, indexing='ij')
      X = tf.scatter_nd(tf.stack(grid, -1), X, shape)

    # Prepare padding info
    if self.padding == "VALID":
      h_pad = 2 * [self.dilations[0] * (self.patch_shape[0] - 1)]
      w_pad = 2 * [self.dilations[1] * (self.patch_shape[1] - 1)]
    elif self.padding == "SAME":
      height_out, width_out = self.get_spatial_out(spatial_in)
      extra = height_out - X.shape[-3]
      h_pad = list(map(lambda x: tf.cast(x, tf.int64),
                       (tf.math.ceil(0.5 * extra), tf.math.floor(0.5 * extra))))

      extra = width_out - X.shape[-2]
      w_pad = list(map(lambda x: tf.cast(x, tf.int64),
                       (tf.math.ceil(0.5 * extra), tf.math.floor(0.5 * extra))))

    # Extract (flipped) image patches
    X_pad = tf.pad(X, [2 * [0], h_pad, w_pad, 2 * [0]])
    patches = tf.image.extract_patches(images=X_pad,
                                       sizes=[1] + self.patch_shape + [1],
                                       strides=[1, 1, 1, 1],
                                       rates=[1] + self.dilations + [1],
                                       padding=self.padding)

    if full_spatial:
      output_shape = list(X.shape[:-3]) + list(patches.shape[-3:])
    else:
      output_shape = list(X.shape[:-3]) + [-1, patches.shape[-1]]

    return tf.reshape(tf.reverse(  # reverse channel-wise patches and reshape
              tf.reshape(patches, list(patches.shape[:-1]) + [-1, X.shape[-1]]),
              axis=[-2]), output_shape)

  def get_spatial_out(self,
                      spatial_in: List = None,
                      filter_shape: List = None,
                      strides: List = None,
                      padding: str = None,
                      dilations: List = None) -> List:

    if spatial_in is None:
      spatial_in = self.image_shape

    if filter_shape is None:
      filter_shape = self.patch_shape
    else:
      assert len(filter_shape) == 2

    if strides is None:
      strides = self.strides

    if padding is None:
      padding = self.padding

    if dilations is None:
      dilations = self.dilations

    return [deconv_output_length(input_length=spatial_in[i],
                                 filter_size=filter_shape[i],
                                 stride=strides[i],
                                 padding=padding.lower(),
                                 dilation=dilations[i]) for i in range(2)]


class DepthwiseConv2d(Conv2d):
  def __init__(self,
               kernel: kernels.Kernel,
               image_shape: List,
               patch_shape: List,
               channels_in: int = 1,
               channels_out: int = 1,
               weights: TensorType = "default",
               strides: List = None,
               padding: str = "VALID",
               dilations: List = None,
               data_format: str = "NHWC",
               **kwargs):

    strides = list((1, 1) if strides is None else strides)
    dilations = list((1, 1) if dilations is None else dilations)
    if strides != [1, 1] and dilations != [1, 1]:
      warn(f"{self.__class__} does not pass unit tests when strides != [1, 1]"
           f"  and dilations != [1, 1] simultaneously.")

    if isinstance(weights, str) and weights == "default":  # TODO: improve me
      spatial_out = self.get_spatial_out(spatial_in=image_shape,
                                         filter_shape=patch_shape,
                                         strides=strides,
                                         padding=padding,
                                         dilations=dilations)

      weights = tf.ones([tf.reduce_prod(spatial_out), channels_in],
                        dtype=default_float())

    super().__init__(kernel=kernel,
                     image_shape=image_shape,
                     patch_shape=patch_shape,
                     channels_in=channels_in,
                     channels_out=channels_out,
                     weights=weights,
                     strides=strides,
                     padding=padding,
                     dilations=dilations,
                     data_format=data_format)

  def K(self, X: tf.Tensor, X2: tf.Tensor = None, full_spatial: bool = False):
    P = self.get_patches(X, full_spatial=full_spatial)
    P2 = P if X2 is None else self.get_patches(X2, full_spatial=full_spatial)

    # TODO: Temporary hack, use of self.kernel should be deprecated
    K = move_axis(
          tf.linalg.diag_part(
            move_axis(self.kernel.K(P, P2), P.shape.ndims - 2, -2)), -1, 0)

    if full_spatial:
      return K  # [channels_in, N, H1, W1, N2, H2, W2]

    # At this point, shape(K) = [N, num_patches, N2, num_patches]
    if self.weights is None:
      return tf.reduce_mean(K, axis=[0, -3, -1])

    K = batch_tensordot(K, self.weights, axes=[-1, 0], batch_axes=[0, 1])
    K = batch_tensordot(K, self.weights, axes=[-2, 0], batch_axes=[0, 1])
    return tf.reduce_mean(K, axis=0)

  def K_diag(self, X: tf.Tensor, full_spatial: bool = False):
    raise NotImplementedError

    P = self.get_patches(X, full_spatial=full_spatial)
    K = tf.reduce_mean(self.kernel.K(P), axis=-2)  # average over channels
    if full_spatial:
      return K  # [num_channels, N, H1, W1, H1, W1]

    # At this point, K has shape # [num_channels, N, num_patches, num_patches]
    if self.weights is None:
      return tf.reduce_mean(K, axis=[-2, -1])

    K = batch_tensordot(K, self.weights, axes=[-1, 0], batch_axes=[0, 1])
    K = batch_tensordot(K, self.weights, axes=[-1, 0], batch_axes=[0, 1])
    return tf.reduce_mean(K, axis=0)

  def convolve(self,
               input,
               filters,
               strides: List = None,
               padding: str = None,
               dilations: List = None,
               data_format: str = None):

    if strides is None:
      strides = self.strides

    if padding is None:
      padding = self.padding

    if dilations is None:
      dilations = self.dilations

    if data_format is None:
      data_format = self.data_format

    return tf.nn.depthwise_conv2d(input=input,
                                  filter=filters,
                                  strides=[1] + strides + [1],
                                  padding=padding,
                                  dilations=dilations,
                                  data_format=data_format)

  def get_patches(self, X: TensorType, full_spatial: bool = False):
    """
    Returns the patches used by a 2d depthwise convolution.
    """
    patches = super().get_patches(X, full_spatial=full_spatial)
    channels_in = X.shape[-3 if self.data_format == "NCHW" else -1]
    depthwise_patches = tf.reshape(patches,
                                   list(patches.shape[:-1]) + [-1, channels_in])
    return move_axis(depthwise_patches, -2, -1)

  def get_shape_out(self,
                    shape_in: List,
                    filter_shape: List,
                    strides: List = None,
                    dilations: List = None,
                    data_format: str = None) -> List:

    if data_format is None:
      data_format = self.data_format

    if data_format == "NHWC":
      *batch, height, width, _ = list(shape_in)
    else:
      *batch, _, height, width = list(shape_in)

    spatial_out = self.get_spatial_out(spatial_in=[height, width],
                                       filter_shape=filter_shape[:2],
                                       strides=strides,
                                       dilations=dilations)

    nhwc_out = batch + spatial_out + [filter_shape[-2] * filter_shape[-1]]
    return conv_ops.reformat_shape(nhwc_out, "NHWC", data_format)
