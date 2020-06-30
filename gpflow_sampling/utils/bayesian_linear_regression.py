#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==============================================
#                                       Preamble
# ==============================================
# ---- Imports
import tensorflow as tf
from typing import *
from tensorflow_probability.python import distributions as tfd
from tensorflow_probability.python.bijectors import CholeskyToInvCholesky
from gpflow.config import default_float
from gpflow.models.model import BayesianModel
from gpflow.likelihoods import Gaussian as GaussianLikelihood
from gpflow_sampling.utils.linalg_ops import parallel_solve, jitter_cholesky

# ---- Exports
__all__ = ('BayesianLinearRegression',)


# ==============================================
#                     bayesian_linear_regression
# ==============================================
class BayesianLinearRegression(BayesianModel):
  def __init__(self,
               prior: tfd.MultivariateNormalDiag,
               likelihood: GaussianLikelihood,
               basis: Callable = None,
               **kwargs):

    assert isinstance(prior, tfd.MultivariateNormalDiag), NotImplementedError
    assert isinstance(likelihood, GaussianLikelihood), NotImplementedError
    super().__init__(**kwargs)
    self.likelihood = likelihood
    self.prior = prior
    self.basis = basis

  def maximum_log_likelihood_objective(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    '''
    [!] Improve me by properly integrating with GPflow's likelihood classes.
    '''
    x, y = self.preprocess_data(data)
    if isinstance(self.prior, tfd.MultivariateNormalDiag):
      var_w = self.prior.variance()
      covar_f = tf.matmul(x, var_w * x, transpose_b=True)
    elif isinstance(self.prior, tfd.MultivariateNormalLinearOperator):
      covar_w = self.prior.covariance()
      covar_f = tf.matmul(x, tf.matmul(x, covar_w), transpose_b=True)
    else:
      raise NotImplementedError

    eye = tf.eye(covar_f.shape[-1], dtype=covar_f.dtype)
    covar_y = covar_f + self.likelihood.variance * eye
    scale_tril = jitter_cholesky(covar_y)
    distrib = tfd.MultivariateNormalTriL(scale_tril=scale_tril)
    return tf.reduce_sum(distrib.log_prob(tf.squeeze(y,  axis=-1)))

  def predict_f(self,
                predict_at: tf.Tensor,
                data: Tuple[tf.Tensor, tf.Tensor] = None,
                w_distrib: tfd.Distribution = None,
                **kwargs) -> tfd.Distribution:

    assert (data is None) ^ (w_distrib is None)
    if w_distrib is None:
      return self._predict_f_from_data(predict_at, data, **kwargs)
    return self._predict_f_from_weights(predict_at, w_distrib, **kwargs)

  def _predict_f_from_data(self,
                           x: tf.Tensor,
                           data: Tuple[tf.Tensor, tf.Tensor],
                           full_cov: bool = False,
                           use_weight_space: bool = None,
                           **kwargs) -> tfd.Distribution:
    """
    Solve for the predictive posterior at a set of query locations
    given observations (X, Y).
    """
    x_train, y_train = self.preprocess_data(data)
    x_eval = x if (self.basis is None) else self.basis(x)
    proj = tf.matmul(x_train, y_train, transpose_a=True)
    eyeN = tf.eye(x_train.shape[-2], dtype=x_train.dtype)  # num. observations
    weight_var = self.prior.variance()
    noise_precis = tf.math.reciprocal(self.likelihood.variance)
    if use_weight_space is None:
      use_weight_space = x_train.shape[-2] >= x_train.shape[-1]

    def solve_weight_space() -> tfd.Distribution:
      d_by_d = tf.matmul(x_train, x_train, transpose_a=True)
      precis = tf.linalg.diag(1/weight_var) + noise_precis * d_by_d
      sqprec = jitter_cholesky(precis)
      solv_y = parallel_solve(tf.linalg.triangular_solve, sqprec,
                              proj, lower=True)
      solv_x = parallel_solve(tf.linalg.triangular_solve, sqprec,
                              tf.linalg.matrix_transpose(x_eval), lower=True)

      loc = noise_precis * tf.matmul(solv_x, solv_y, transpose_a=1)
      if self.mean_function is not None:
        loc += self.mean_function(x)

      if full_cov:
        scale_tril = jitter_cholesky(tf.matmul(solv_x, solv_x, transpose_a=True))
        return tfd.MultivariateNormalTriL(loc=tf.squeeze(loc, axis=-1),
                                          scale_tril=scale_tril)
      else:
        scale_diag = tf.sqrt(tf.reduce_sum(tf.square(solv_x), axis=-2))
        return tfd.MultivariateNormalDiag(loc=tf.squeeze(loc, axis=-1),
                                          scale_diag=scale_diag)

    def solve_function_space() -> tfd.Distribution:
      n_by_n = tf.matmul(x_train,
                         weight_var[..., None, :] * x_train,
                         transpose_b=True)
      precis = (1/noise_precis) * eyeN + n_by_n
      sqprec = jitter_cholesky(precis)
      solv_x = parallel_solve(tf.linalg.triangular_solve,
                              sqprec, x_train, lower=True)
      w_covar = solv_x * weight_var[..., None, :]  # scratch variable
      w_covar = tf.linalg.diag(weight_var) \
                - tf.matmul(w_covar, w_covar, transpose_a=True)

      loc = noise_precis * tf.matmul(x_eval, tf.matmul(w_covar, proj))
      covar = tf.matmul(x_eval, tf.matmul(w_covar, x_eval, transpose_b=True))
      if full_cov:
        scale_tril = jitter_cholesky(covar)
        return tfd.MultivariateNormalTriL(loc=tf.squeeze(loc, axis=-1),
                                          scale_tril=scale_tril)
      else:
        scale_diag = tf.sqrt(tf.linalg.diag_part(covar))
        return tfd.MultivariateNormalDiag(loc=tf.squeeze(loc, axis=-1),
                                          scale_diag=scale_diag)

    return solve_weight_space() if use_weight_space else solve_function_space()

  def _predict_f_from_weights(self,
                              x: tf.Tensor,
                              w_distrib: tfd.Distribution,
                              full_cov: bool = False,
                              **kwargs) -> tfd.Distribution:
    """
    Solve for the predictive posterior at a set of query locations
    given a distribution over weights.
    """
    _x = x if (self.basis is None) else self.basis(x)
    loc = tf.reduce_sum(_x * w_distrib.loc, axis=-1)
    proj = tf.matmul(_x, w_distrib.scale)
    if full_cov:
      cov_x = tf.matmul(proj, proj, transpose_b=True)
      chol_x = jitter_cholesky(cov_x)
      distrib = tfd.MultivariateNormalTriL(loc=loc, scale_tril=chol_x)
    else:
      var_x = tf.reduce_sum(tf.square(proj), axis=-1)
      distrib = tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.sqrt(var_x))
    return distrib

  def predict_f_samples(self,
                        predict_at: tf.Tensor,
                        sample_shape: Union[tuple, list],
                        data: Tuple[tf.Tensor, tf.Tensor] = None,
                        w_distrib: tfd.Distribution = None,
                        full_cov: bool = False,
                        use_weight_space: bool = None,
                        **kwargs) -> tf.Tensor:
    """
    Generate samples from random function $f$ evaluated at test locations.
    """
    if use_weight_space is None:
      if self.basis is None:
        num_weights = predict_at.shape[-1]
      else:
        num_weights = self.basis.units
      num_predict = predict_at.shape[-2] if full_cov else 1
      use_weight_space = num_predict > num_weights

    # Sample via weight-space representation
    if use_weight_space:
      if w_distrib is None:
        w_samples = self.predict_w_samples(sample_shape, data, **kwargs)
      else:
        w_samples = w_distrib.sample(sample_shape)

      if self.basis is None:
        x = predict_at
      else:
        x = self.basis(predict_at)

      f_samples = tf.matmul(w_samples, x, transpose_b=True)
      return f_samples

    # Sample via function-space representation
    f_distrib = self.predict_f(predict_at,
                               data=data,
                               w_distrib=w_distrib,
                               full_cov=full_cov,
                               **kwargs)

    return f_distrib.sample(sample_shape)

  def predict_w(self,
                data: Tuple[tf.Tensor, tf.Tensor],
                use_weight_space: bool = None) -> tfd.Distribution:
    """
    Solve for the optimal posterior distribution of weights.
    """
    x, y = self.preprocess_data(data)
    proj = tf.matmul(x, y, transpose_a=True)
    eyeN = tf.eye(x.shape[-2], dtype=x.dtype)  # num. observations
    weight_var = self.prior.variance()
    noise_precis = tf.math.reciprocal(self.likelihood.variance)
    if use_weight_space is None:
      use_weight_space = x.shape[-2] >= x.shape[-1]

    def solve_weight_space() -> tfd.Distribution:
      d_by_d = tf.matmul(x, x, transpose_a=True)
      precis = tf.linalg.diag(1/weight_var) + noise_precis * d_by_d
      sqprec = jitter_cholesky(precis)
      means = noise_precis * parallel_solve(tf.linalg.cholesky_solve,
                                            sqprec,
                                            proj)
      scale_tril = CholeskyToInvCholesky()(sqprec)  # [!] improve me
      return tfd.MultivariateNormalTriL(loc=tf.squeeze(means, -1),
                                        scale_tril=scale_tril)

    def solve_function_space() -> tfd.Distribution:
      n_by_n = tf.matmul(x, weight_var[..., None, :] * x, transpose_b=True)
      precis = (1/noise_precis) * eyeN + n_by_n
      sqprec = jitter_cholesky(precis)
      solv = parallel_solve(tf.linalg.triangular_solve, sqprec, x, lower=True)

      covar = solv * weight_var[..., None, :]  # scratch variable
      covar = tf.linalg.diag(weight_var) \
                - tf.matmul(covar, covar, transpose_a=True)
      means = noise_precis * tf.matmul(covar, proj)
      scale_tril = jitter_cholesky(covar)
      return tfd.MultivariateNormalTriL(loc=tf.squeeze(means, -1),
                                        scale_tril=scale_tril)
    return solve_weight_space() if use_weight_space else solve_function_space()

  def predict_w_samples(self,
                        sample_shape: Union[tuple, list],
                        data: Tuple[tf.Tensor, tf.Tensor],
                        **kwargs) -> tf.Tensor:
    """
    Generate samples from conditional distribution of weights $w$.
    """
    num_datum = data[0].shape[-2]
    if self.basis is None:
      num_weights = data[0].shape[-1]
    else:
      num_weights = self.basis.units

    if num_datum < num_weights:
      w_samples = self.prior.sample(sample_shape=sample_shape)
      return self.update_w_samples(w_distrib=self.prior,
                                   w_samples=w_samples,
                                   data=data,
                                   **kwargs)
    else:
      w_distrib = self.predict_w(data=data, **kwargs)
      return w_distrib.sample(sample_shape=sample_shape)

  def update_w_samples(self,
                       w_distrib: tfd.Distribution,
                       w_samples: tf.Tensor,
                       data: Tuple[tf.Tensor, tf.Tensor],
                       noisy: bool = True) -> tf.Tensor:
    """
    Use Matheron's rule to directly condition weights drawn from the prior
    on observations (x, y).
    """
    x, y = self.preprocess_data(data)
    yhat = tf.matmul(x, w_samples[..., None, :], transpose_b=True)
    yhat += tf.sqrt(self.likelihood.variance) * tf.random.normal(shape=yhat.shape,
                                                                 dtype=yhat.dtype)
    resid = y - yhat  # implicit negative
    noise_var = self.likelihood.variance
    if isinstance(w_distrib, tfd.MultivariateNormalDiag):
      D = w_distrib.stddev()
      B = D[..., None, :] * x
      precis = tf.matmul(B, B, transpose_b=True)
      if noisy:
        precis += noise_var * tf.eye(precis.shape[-2], dtype=precis.dtype)
      sqprec = jitter_cholesky(precis)
      solv = parallel_solve(tf.linalg.cholesky_solve, sqprec, resid)
      update = D * tf.squeeze(tf.matmul(solv, B, transpose_a=True), -2)
    elif isinstance(w_distrib, tfd.MultivariateNormalTriL):
      L = w_distrib.scale
      B = tf.matmul(x, L)
      precis = tf.matmul(B, B, transpose_b=True)
      if noisy:
        precis += noise_var * tf.eye(precis.shape[-2], dtype=precis.dtype)
      sqprec = jitter_cholesky(precis)
      solv = parallel_solve(tf.linalg.cholesky_solve, sqprec, resid)
      update = tf.matmul(tf.matmul(tf.squeeze(solv, -1), B), L, transpose_b=True)
    else:
      raise NotImplementedError

    return w_samples + update

  def preprocess_data(self,
                      data: Tuple[tf.Tensor, tf.Tensor],
                      dtype: Any = None,
                      **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Convenience method for converting (x, y) pairs into model format.
    """
    if dtype is None:
      dtype = default_float()

    x, y = map(lambda src: tf.convert_to_tensor(src, dtype=dtype), data)
    if self.basis is not None:
      x = self.basis(x, **kwargs)
    return x, y
