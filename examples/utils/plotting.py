#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import seaborn as sns
import matplotlib.colors as mc
import colorsys

from typing import *
from numpy import squeeze, atleast_1d, linspace
from contextlib import contextmanager


# ---- Exports
__all__ = (
  'AttrDict',
  'ObjectStyle',
  'default_style',
  'set_default_style',
  'set_default_rcParams',
  'set_temporary_rcParams',
  'plot_line',
  'plot_scatter',
  'plot_fill',
  'plot_arrow',
  'plot_axvlines',
  'format_axes',
  'default_ax',
  'set_default_ax',
  'set_temporary_ax',
  'adjust_lightness',
)


# ==============================================
#                                       plotting
# ==============================================
class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self


class ObjectStyle(AttrDict):
  def __init__(self, label: Any, color: Any, **kwargs):
    super().__init__(label=label, color=color, **kwargs)

  def omit(self, *keys: Tuple) -> AttrDict:
    return AttrDict(filter(lambda kv: kv[0] not in keys, self.items()))


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self


class ObjectStyle(AttrDict):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def omit(self, *keys: Tuple) -> AttrDict:
    return AttrDict(filter(lambda kv: kv[0] not in keys, self.items()))


def default_style():
  return _DEFAULT_STYLE


def set_default_style(style: AttrDict):
  global _DEFAULT_STYLE
  _DEFAULT_STYLE = style


def set_default_rcParams(plt, fontsize=12, usetex=True):
  if usetex:
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}\usepackage{amsfonts}')
    plt.rc('text', usetex=usetex)

  plt.rc('font', family='serif', size=fontsize)
  plt.rc('figure', dpi=256)


@contextmanager
def set_temporary_rcParams(plt, params: dict):
  cache = {key: plt.rcParams[key] for key in params}

  plt.rcParams.update(params)
  try:
    yield plt.rcParams
  finally:
    plt.rcParams.update(cache)


def default_ax():
  return _DEFAULT_AX


def set_default_ax(ax):
  global _DEFAULT_AX
  _DEFAULT_AX = ax


@contextmanager
def set_temporary_ax(ax):
  _ax = default_ax()
  set_default_ax(ax)
  try:
    yield default_ax()
  finally:
    set_default_ax(_ax)


def plot_line(x, y, *args, zorder=10, ax=None, alpha=0.9, linewidth=1.5, **kwargs):
  if ax is None:
      ax = default_ax()
  data = map(lambda arr: atleast_1d(squeeze(arr)), (x, y))
  return ax.plot(*data, *args, zorder=zorder, alpha=alpha, linewidth=linewidth, **kwargs)


def plot_scatter(x, y, *args, zorder=1000, ax=None, alpha=0.9, legend=False, **kwargs):
  if ax is None:
      ax = default_ax()
  data = map(lambda arr: atleast_1d(squeeze(arr)), (x, y))
  return sns.scatterplot(*data, *args,
                         ax=ax,
                         zorder=zorder,
                         alpha=alpha,
                         legend=legend,
                         **kwargs)


def plot_fill(x, lower, upper, *args, ax=None, zorder=1, alpha=0.15, **kwargs):
  if ax is None:
    ax = default_ax()
  data = map(lambda arr: squeeze(arr), (x, lower, upper))
  return ax.fill_between(*data, *args, zorder=zorder, alpha=alpha, **kwargs)


def plot_arrow(x, y, dx, dy, ax=None, zorder=10, alpha=0.9, head_width=0.1,
               head_length=0.2, length_includes_head=True, **kwargs):
  if ax is None:
      ax = default_ax()
  return ax.arrow(x, y, dx, dy,
                  zorder=zorder,
                  alpha=alpha,
                  head_width=head_width,
                  head_length=head_length,
                  length_includes_head=length_includes_head,
                  **kwargs)


def plot_axvlines(x, ax=None, zorder=1, **kwargs):
  if ax is None:
    ax = default_ax()

  axvlines = []
  for xval in atleast_1d(squeeze(x)):
    axvlines.append(ax.axvline(xval, zorder=zorder, **kwargs))
  return axvlines


def format_axes(ax=None,
                xlim=None,
                ylim=None,
                num_major=3,
                num_minor=5,
                despine=False,
                grid=None):

  if ax is None:
      ax = default_ax()

  if xlim is not None:
      ax.set_xlim(xlim)

  if ylim is not None:
      ax.set_ylim(ylim)

  ax.set_xticks(linspace(*ax.get_xlim(), num_major), minor=False)
  ax.set_xticks(linspace(*ax.get_xlim(), num_minor), minor=True)

  ax.set_yticks(linspace(*ax.get_ylim(), num_major), minor=False)
  ax.set_yticks(linspace(*ax.get_ylim(), num_minor), minor=True)

  if despine:
    sns.despine(ax=ax)

  if grid is not None:
      ax.grid(color='silver',
              linestyle='-',
              which=grid,
              linewidth=0.5,
              alpha=1/3,
              zorder=1)


def adjust_lightness(color, amount=0.5):
  """
  Returns a lighter/darker version of the provided color
  """
  try:
    c = mc.cnames[color]
  except:
    c = color
  c = colorsys.rgb_to_hls(*mc.to_rgb(c))
  return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


_DEFAULT_AX = None
_DEFAULT_STYLE = AttrDict(data=ObjectStyle(color='black'),
                          loc_scale=ObjectStyle(color='tab:green'),
                          rff=ObjectStyle(color='tab:orange'),
                          sparse=ObjectStyle(color='tab:purple'),
                          decoupled=ObjectStyle(color='tab:blue'))
