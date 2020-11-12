#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from typing import List
from gpflow.base import TensorType
from gpflow_sampling.utils.array_ops import move_axis


# ---- Exports
__all__ = (
  'reformat_shape',
  'reformat_data',
)


# ==============================================
#                                       conv_ops
# ==============================================
def reformat_shape(shape: List,
                   input_format: str,
                   output_format: str) -> List:
  """
  Helper method for shape data between NHWC and NCHW formats.
  """
  if input_format == output_format:
    return shape  # noop

  if input_format == 'NHWC':
    assert output_format == "NCHW"
    return shape[:-3] + shape[-1:] + shape[-3: -1]

  if input_format == 'NCHW':
    assert output_format == "NHWC"
    return shape[:-3] + shape[-2:] + [shape[-3]]

  raise NotImplementedError


def reformat_data(x: TensorType,
                  format_in: str,
                  format_out: str):
  """
  Helper method for converting image data between NHWC and NCHW formats.
  """
  if format_in == format_out:
    return x  # noop

  if format_in == "NHWC":
    assert format_out == "NCHW"
    return move_axis(x, -1, -3)

  if format_in == "NCHW":
    assert format_out == "NHWC"
    return move_axis(x, -3, -1)

  raise NotImplementedError
