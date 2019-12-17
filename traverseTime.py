from absl import app
from absl import logging

import jax as jax
import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers

import sys
import os
import numpy as onp
import matplotlib.pyplot as plt

from modules.flags import get_flags
from modules.data import get_data
from modules.parameters import get_parameters
from modules.diffraction import *
from modules.losses import earth_movers_distance

FLAGS = get_flags()


def simulate_exp(init_geo, delta_geo, q, r, scat_weights):

  dists   = dist_metric_td(init_geo + delta_geo)
  mm_diffraction, _, _ = diffraction(
      dists, q, scat_weights)
  pair_corr = calc_pair_corr(q, r, mm_diffraction, 3.5)

  return mm_diffraction, pair_corr


def diffraction_loss(diffraction_fit, diffraction_truth):

  return np.mean((diffraction_fit - diffraction_truth)**2)


def pairCorr_loss(pairCorr_fit, pairCorr_truth):

  #return earth_movers_distance(pairCorr_fit, pairCorr_truth)
  return np.mean((pairCorr_fit - pairCorr_truth)**2)


def velocity_loss(init_geo, delta_geo, v_scale, dt):

  dists = dist_metric_td(init_geo + delta_geo)
  velocity = (dists[1:] - dists[:-1])/(v_scale*dt)
  return np.sum(velocity**2)


def init_zero_loss(delta_geo):

  return np.sum(np.abs(delta_geo[0,:,:]))


def calculate_losses(
    params, init_geo,
    q, r, scat_weights,
    v_scale, dt,
    diffraction_truth, pairCorr_truth):

  delta_geo = params
    
  diffraction_fit, pairCorr_fit = simulate_exp(init_geo, delta_geo, q, r, scat_weights)

  diffraction_loss_value  = diffraction_loss(diffraction_fit, diffraction_truth)
  pairCorr_loss_value     = pairCorr_loss(pairCorr_fit, pairCorr_truth)
  velocity_loss_value     = velocity_loss(init_geo, delta_geo, v_scale, dt)
  init_zero_loss_value    = init_zero_loss(delta_geo)

  return diffraction_loss_value,\
      pairCorr_loss_value,\
      velocity_loss_value,\
      init_zero_loss_value


def loss(
    params, init_geo,
    q, r, scat_weights, 
    v_scale, dt,
    diffraction_truth, pairCorr_truth):

  loss_diffraction, loss_pairCorr, loss_velocity, loss_init_zero =\
      calculate_losses(
          params, init_geo,
          q, r, scat_weights,
          v_scale, dt,
          diffraction_truth, pairCorr_truth)

  return FLAGS.scale_diffraction_loss*loss_diffraction\
      + FLAGS.scale_pairCorr_loss*loss_pairCorr\
      + FLAGS.scale_init_zero_loss*loss_init_zero\
      + FLAGS.scale_velocity_loss*loss_velocity



def main(argv):
  logging.info("Running main")

  print(FLAGS.leFlag)

  rng = random.PRNGKey(0)
  diffraction_params = get_parameters()

  step_history  = []
  loss_history  = []
  dists_history = []
  angle_history = []

  ######################
  #####  Get Data  #####
  ######################

  init_geo, dists_data, angles_data, mmd_data, pairCorr_data = get_data(FLAGS)
  print(init_geo, init_geo.shape)
  
  q = get_Q(diffraction_params)
  r = get_R(diffraction_params)
  scat_amps, scat_weights = get_scattering_amps(diffraction_params, q)


  #############################
  #####  Setup Optimizer  #####
  #############################

  #opt_init, opt_update, get_params = optimizers.sgd(1)
  opt_init, opt_update, get_params = optimizers.adam(1e-4)

  delta_geo = np.zeros((dists_data.shape[0], init_geo.shape[-2], init_geo.shape[-1]))
  opt_state = opt_init(delta_geo)

  @jit
  def update(
      stp, opt_state, init_geo,
      q, r, scat_weights, 
      v_scale, dt,
      diffraction_truth, pairCorr_truth):

    params = get_params(opt_state)
    return opt_update(
        stp,
        grad(loss)(
            params, init_geo,
            q, r, scat_weights,
            v_scale, dt,
            diffraction_truth, pairCorr_truth),
        opt_state)


  @jit
  def asses_fit(
      opt_state, init_geo,
      q, r, scat_weights,
      v_scale, dt,
      diffraction_truth, pairCorr_truth):
    
    params = get_params(opt_state)
    loss_diffraction, loss_pairCorr, loss_velocity, loss_init_zero =\
      calculate_losses(
          params, init_geo,
          q, r, scat_weights,
          v_scale, dt,
          diffraction_truth, pairCorr_truth)

    loss_diffraction  *= FLAGS.scale_diffraction_loss
    loss_pairCorr     *= FLAGS.scale_pairCorr_loss
    loss_velocity     *= FLAGS.scale_velocity_loss
    loss_init_zero    *= FLAGS.scale_init_zero_loss
    losses = [loss_diffraction, loss_pairCorr, loss_velocity, loss_init_zero]

    return np.sum(losses), losses
 

  def append_history(stp, losses, delta_geo, init_geo):
    dists   = dist_metric_td(init_geo + delta_geo)
    angles  = angle_metric_td(init_geo + delta_geo)

    step_history.append(stp)
    loss_history.append(np.expand_dims(np.array(losses), 0))
    dists_history.append(np.expand_dims(dists, 0))
    #angle_history.append(np.expand_dims(angles, 0))

 

  for stp in range(15000):
    print(stp)
    params = get_params(opt_state)
    opt_state = update(
        stp, opt_state, init_geo,
        q, r, scat_weights,
        FLAGS.velocity_scale, diffraction_params["dt"],
        mmd_data, pairCorr_data)

    if stp % FLAGS.log_every == 0:
      loss_val, losses = asses_fit(
        opt_state, init_geo,
        q, r, scat_weights,
        FLAGS.velocity_scale, diffraction_params["dt"],
        mmd_data, pairCorr_data)

      append_history(stp, losses, params, init_geo)

      log_info = "Step {}\tLoss {} / {}".format(stp, loss_val, losses)
      logging.info(log_info)

  #####  Save Fitting History  #####
  save_dir = os.path.join(FLAGS.history_dir, FLAGS.molecule)
  with open(os.path.join(save_dir, "step_history.npy"), "wb") as file:
    np.save(file, np.array(step_history))

  with open(os.path.join(save_dir, "loss_history.npy"), "wb") as file:
    np.save(file, np.concatenate(loss_history))

  with open(os.path.join(save_dir, "dists_history.npy"), "wb") as file:
    np.save(file, np.concatenate(dists_history))

  #with open(os.path.join(save_dir, "angle_history.npy"), "wb") as file:
  #  np.save(file, np.concatenate(angle_history))




if __name__ == "__main__":

  app.run(main)
