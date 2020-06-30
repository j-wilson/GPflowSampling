#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
"""
TODO: Refactor use of bijectors to be more efficient/explicit
"""

# ---- Imports
import tensorflow as tf

from abc import abstractmethod
from typing import *
from gpflow.config import default_float
from tensorflow_probability.python import bijectors as tfb

# ---- Exports
__all__ = (
  'DynamicalSystem',
  'ModelBasedSystem',
  'LotkaVolterra',
  'FitzHughNagumo',
  'HodgkinHuxley',
)


# ==============================================
#                              dynamical_systems
# ==============================================
class DynamicalSystem:
  def __init__(self,
               noise_func: Callable = None,
               controllable: bool = False,
               bijector: tfb.Bijector = None):

    if bijector is None:
      bijector = self.default_bijector()

    self.bijector = bijector
    self.noise_func = noise_func  # diffusion
    self.controllable = controllable

  @abstractmethod
  def drift_func(self,
                 x: tf.Tensor,
                 u: tf.Tensor = None,
                 **kwargs) -> tf.Tensor:
    """
    Deterministic component of transition operator.
    """
    raise NotImplementedError

  def forward(self,
              x: tf.Tensor,
              t: tf.Tensor,
              dt: tf.Tensor,
              noisy: bool = True,
              **kwargs) -> tf.Tensor:

    _x = self.bijector.forward(x)
    _dx = self.drift_func(_x, t=t, dt=dt, **kwargs)
    if noisy and self.noise_func is not None:
      _dx += self.noise_func(_x, t=t, dt=dt)
    return self.bijector.inverse(_dx)

  def euler_step(self,
                 x: tf.Tensor,
                 t: tf.Tensor,
                 dt: tf.Tensor,
                 noisy: bool = True,
                 **kwargs) -> tf.Tensor:
    """
    Euler-Maruyama forward step
    """
    _x = self.bijector.forward(x)
    _dx = self.drift_func(_x, t=t, dt=dt, **kwargs)
    if noisy and self.noise_func is not None:
      _dx += self.noise_func(_x, t=t, dt=dt)
    return self.bijector.inverse(_x + dt * _dx)

  def unroll(self,
             x0: tf.Tensor,
             time_steps: tf.Tensor,
             control_func: Callable = None,
             **kwargs) -> tf.Tensor:

    def closure(x, t_dt):
      t, dt = t_dt
      if control_func is None:
        u = None
      else:
        u = control_func(x=x, t=t, dt=dt)
      return self.euler_step(x, t=t, dt=dt, u=u, **kwargs)

    dts = time_steps[1:] - time_steps[:-1]
    return tf.scan(closure, [time_steps[:-1], dts], initializer=x0)

  def check_control(self, u: tf.Tensor = None):
    if not self.controllable:
      assert u is None, \
        ValueError(f'{self.__class__} instance is not controllable.')

  @classmethod
  def default_bijector(cls, **kwargs) -> tfb.Bijector:
    return tfb.Identity()


class ModelBasedSystem(DynamicalSystem):
  def __init__(self, model, **kwargs):
    super().__init__(**kwargs)
    self.model = model

  def drift_func(self,
                 x: tf.Tensor,
                 u: tf.Tensor = None,
                 **kwargs) -> tf.Tensor:

    self.check_control(u)
    if self.controllable:  # broadcast and concatenate (x, u)
      shape_u = list(u.shape)
      shape_x = list(x.shape)
      ndims_u = len(shape_u)
      ndims_x = len(shape_x)
      if ndims_x > ndims_u:
        shape = shape_x[:ndims_x - ndims_u] + shape_u
        u = tf.broadcast_to(u, shape)
      elif ndims_u > ndims_x:
        shape = shape_u[:ndims_u - ndims_x] + shape_x
        x = tf.broadcast_to(x, shape)
      inputs = tf.concat([x, u], axis=-1)
    else:
      inputs = x
    return self.model(inputs, **kwargs)

  def forward(self,
              x: tf.Tensor,
              t: tf.Tensor,
              dt: tf.Tensor,
              noisy: bool = True,
              **kwargs) -> tf.Tensor:

    dx = self.drift_func(x, t=t, dt=dt, **kwargs)
    _dx = self.bijector.forward(dx)
    if noisy and self.noise_func is not None:
      _x = self.bijector.forward(x)
      _dx += self.noise_func(_x, t=t, dt=dt)
    return self.bijector.inverse(_dx)

  def euler_step(self,
                 x: tf.Tensor,
                 t: tf.Tensor,
                 dt: tf.Tensor,
                 noisy: bool = True,
                 **kwargs) -> tf.Tensor:
    """
    Euler-Maruyama forward step
    """
    dx = self.drift_func(x, t=t, dt=dt, **kwargs)
    _x = self.bijector.forward(x)
    _dx = self.bijector.forward(dx)
    if noisy and self.noise_func is not None:
      _dx += self.noise_func(_x, t=t, dt=dt)
    return self.bijector.inverse(_x + dt * _dx)

  def check_control(self, u: tf.Tensor = None):
    if self.controllable:
      assert u is not None, \
        ValueError(f'{self.__class__} instance requires control inputs.')
    else:
      assert u is None, \
        ValueError(f'{self.__class__} instance is not controllable.')


class LotkaVolterra(DynamicalSystem):
  def __init__(self,
               alpha: float = 2/3,
               beta: float = 4/3,
               gamma: float = 1.0,
               delta: float = 1.0,
               bijector: tfb.Bijector = None,
               **kwargs):
    """
    Lotka-Volterra (a.k.a. predator-prey) ecological model.
    """
    if bijector is None:
      bijector = self.default_bijector()

    super().__init__(bijector=bijector, controllable=False, **kwargs)
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.delta = delta

  def drift_func(self,
                  x: tf.Tensor,
                  u: tf.Tensor = None,
                  **kwargs) -> tf.Tensor:

    self.check_control(u)
    x0, x1 = tf.split(x, 2, axis=-1)
    x01 = x0 * x1
    dx0 = self.alpha * x0 - self.beta * x01
    dx1 = self.delta * x01 - self.gamma * x1
    return tf.concat([dx0, dx1], axis=-1)

  @classmethod
  def default_bijector(cls, dtype: Any = None, **kwargs) -> tfb.Bijector:
    """
    Linear bijection between $[0, 1]^{2} <--> [0, 4]^{2}$
    """
    if dtype is None:
      dtype = default_float()

    return tfb.Scale(tf.cast(4.0, dtype=dtype))


class FitzHughNagumo(DynamicalSystem):
  def __init__(self,
               alpha: float = 0.75,
               beta: float = 0.75,
               tau: float = 20.0,
               bijector: tfb.Bijector = None,
               **kwargs):
    """
    FitzHugh-Nagumo 2-dimensional simplification of Hodgkin-Huxley
    model neuron.
    """
    if bijector is None:
      bijector = self.default_bijector()

    super().__init__(bijector=bijector, controllable=True, **kwargs)
    self.alpha = alpha
    self.beta = beta
    self.tau = tau

  def drift_func(self,
                  x: tf.Tensor,
                  u: tf.Tensor = None,
                  **kwargs) -> tf.Tensor:

    self.check_control(u)
    v, w = tf.split(x, 2, axis=-1)
    dv = v - tf.pow(v, 3) / 3 - w
    if u is not None:
      dv += u  # control input

    dw = (v + self.alpha - self.beta * w) / self.tau
    return tf.concat([dv, dw], axis=-1)

  @classmethod
  def default_bijector(cls, dtype: Any = None, **kwargs) -> tfb.Bijector:
    """
    Affine bijection between $[[0, 1], [0, 1]] <--> [[-2.5, 2.5], [-1.0, 2.0]]$
    """
    if dtype is None:
      dtype = default_float()
    scale = tfb.Scale(tf.convert_to_tensor([5.0, 3.0], dtype=dtype))
    shift = tfb.Shift(tf.convert_to_tensor([-0.5, -1 / 3], dtype=dtype))
    return tfb.Chain([scale, shift])


class HodgkinHuxley(DynamicalSystem):
  def __init__(self,
               cM: float = 1.0,
               gK: float = 36.0,
               vK: float = -12.0,
               gNa: float = 120.0,
               vNa: float = 115.0,
               gL: float = 0.3,
               vL: float = 10.6,
               bijector: tfb.Bijector = None,
               **kwargs):

    if bijector is None:
      bijector = self.default_bijector()

    super().__init__(bijector=bijector, controllable=True, **kwargs)
    self.cM = cM
    self.gL = gL
    self.vL = vL
    self.gK = gK
    self.vK = vK
    self.gNa = gNa
    self.vNa = vNa

  def drift_func(self,
                  x: tf.Tensor,
                  u: tf.Tensor = None,
                  **kwargs) -> tf.Tensor:

    self.check_control(u)
    v, n, m, h = tf.split(x, 4, axis=-1)
    gL = self.gL/self.cM
    gK = self.gK/self.cM * tf.math.pow(n, 4.0)
    gNa = self.gNa/self.cM * tf.math.pow(m, 3.0) * h

    dv = (0.0 if (u is None) else u/self.cM) \
        + gK * (self.vK - v) \
        + gNa * (self.vNa - v) \
        + gL * (self.vL - v)

    dn = (1 - n) * self.alpha_n(v) - n * self.beta_n(v)
    dm = (1 - m) * self.alpha_m(v) - m * self.beta_m(v)
    dh = (1 - h) * self.alpha_h(v) - h * self.beta_h(v)
    return tf.concat([dv, dn, dm, dh], axis=-1)

  def alpha_n(self, v: tf.Tensor) -> tf.Tensor:
    return tf.divide(0.1 - 0.01 * v, tf.math.expm1(1.0 - 0.1 * v))

  def beta_n(self, v: tf.Tensor) -> tf.Tensor:
    return 0.125 * tf.math.exp(-0.0125 * v)

  def alpha_m(self, v: tf.Tensor) -> tf.Tensor:
    return tf.divide(2.5 - 0.1 * v, tf.math.expm1(2.5 - 0.1 * v))

  def beta_m(self, v: tf.Tensor) -> tf.Tensor:
    return 4.0 * tf.math.exp(-1/18.0 * v)

  def alpha_h(self, v: tf.Tensor) -> tf.Tensor:
    return 0.07 * tf.math.exp(-0.05 * v)

  def beta_h(self, v: tf.Tensor) -> tf.Tensor:
    return tf.math.reciprocal(1.0 + tf.math.exp(3.0 - 0.1 * v))

  @classmethod
  def default_bijector(cls, dtype: Any = None, **kwargs) -> tfb.Bijector:
    """
    Affine bijection between $[[0, 1]]^4 <--> [[-20, 120]] x [[0, 1]]^3$
    """
    if dtype is None:
      dtype = default_float()

    scale = tfb.Scale(tf.convert_to_tensor([140] + 3*[1], dtype=dtype))
    shift = tfb.Shift(tf.convert_to_tensor([-1/7] + 3*[0], dtype=dtype))
    return tfb.Chain([scale, shift])
