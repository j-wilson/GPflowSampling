#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from typing import Optional
from gpflow import inducing_variables
from gpflow.base import TensorData, Parameter
from gpflow.config import default_float
from gpflow_sampling.utils import move_axis


# ---- Exports
__all__ = (
  'InducingImages',
  'SharedInducingImages',
  'DepthwiseInducingImages',
  'SharedDepthwiseInducingImages',
)


# ==============================================
#                             inducing_variables
# ==============================================
class InducingImages(inducing_variables.InducingVariables):
  def __init__(self, images: TensorData, name: Optional[str] = None):
      """
      :param images: initial values of inducing locations in image form.

      The shape of the inducing variables varies by representation:
        - as Z: [M, height * width * channels_in]
        - as images: [M, height, width, channels_in]
        - as patches: [M, height * width * channels_in]
        - as filters: [height, width, channels_in, M]

      TODO:
        - Generalize to allow for inducing image with multiple patches?
        - Work on naming convention? The term 'image' is a bit too general.
          Patch works, however this term  usually refers to a vectorized form
          and (for now) overlaps with GPflow's own inducing class. Alternatives
          include: filter, window, glimpse
      """
      super().__init__(name=name)
      self._images = Parameter(images, dtype=default_float())

  def __len__(self) -> int:
    return self._images.shape[0]

  @property
  def Z(self) -> tf.Tensor:
    return tf.reshape(self._images, [len(self), -1])

  @property
  def as_patches(self) -> tf.Tensor:
    return tf.reshape(self.as_images, [len(self), -1])

  @property
  def as_filters(self) -> tf.Tensor:
    return move_axis(self.as_images, 0, -1)

  @property
  def as_images(self) -> tf.Tensor:
    return tf.convert_to_tensor(self._images, dtype=self._images.dtype)


class SharedInducingImages(InducingImages):
  def __init__(self,
               images: TensorData,
               channels_in: int,
               name: Optional[str] = None):
    """
    :param images: initial values of inducing locations in image form.
    :param channels_in: number of input channels to share across

    Same as <InducingImages> but with the same single-channel inducing
    images shared across all input channels.

    The shape of the inducing variables varies by representation:
      - as Z: [M, height * width]  (new!)
      - as images: [M, height, width, channels_in]
      - as patches [M, channels_in, height * width]
      - as filters: [height, width, channels_in, M]
    """
    assert images.shape.ndims == 4 and images.shape[-1] == 1
    self.channels_in = channels_in
    super().__init__(images, name=name)

  @property
  def as_images(self) -> tf.Tensor:
    return tf.tile(self._images, [1, 1, 1, self.channels_in])


class DepthwiseInducingImages(InducingImages):
  """
  Same as <InducingImages> but for depthwise convolutions.

  The shape of the inducing variables varies by representation:
    - as Z: [M, height * width * channels_in]
    - as images: [M, height, width, channels_in]
    - as patches [M, channels_in, height * width]  (new!)
    - as filters: [height, width, channels_in, M]
  """
  @property
  def as_patches(self) -> tf.Tensor:
    images = self.as_images
    patches = tf.reshape(images, [images.shape[0], -1, images.shape[-1]])
    return tf.transpose(patches, [0, 2, 1])  # [M, channels_in, patch_len]


class SharedDepthwiseInducingImages(SharedInducingImages,
                                    DepthwiseInducingImages):
  def __init__(self,
               images: TensorData,
               channels_in: int,
               name: Optional[str] = None):
    """
    :param images: initial values of inducing locations in image form.
    :param channels_in: number of input channels to share across.
    """
    SharedInducingImages.__init__(self,
                                  name=name,
                                  images=images,
                                  channels_in=channels_in)
