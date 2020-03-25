from absl import app
from absl import logging

import jax as jax
import jax.numpy as np
from jax import jit, grad, random
from jax.experimental import optimizers
from jax_md import minimize

import sys
import os
import glob
import pickle as pl
import numpy as onp
import matplotlib.pyplot as plt
from hashlib import blake2b

from modules.flags.flags_EM import get_flags
from modules.data import get_data
from modules.parameters import get_parameters
from modules.diffraction import * 
from modules.losses import *

FLAGS = get_flags()


def simulate_exp(init_geo, delta_geo, q, r, scat_weights):

  dists   = dist_metric_td(init_geo + delta_geo)
  mm_diffraction, _, _ = diffraction(
      dists, q, scat_weights)
  pair_corr = calc_pair_corr(q, r, mm_diffraction, 3.5)

  return mm_diffraction, pair_corr


def get_dependent_loss_fxns(FLAGS, parameters, init_geo):
  
  def diffraction_loss_EM(diffraction_fit, diffraction_truth):
    return earth_movers_distance(diffraction_fit, diffraction_truth)

  def diffraction_loss_L2(diffraction_fit, diffraction_truth):
    return L2_loss(diffraction_fit, diffraction_truth)

  if False and FLAGS.loss_type == "EM":
    diffraction_loss = diffraction_loss_EM
  elif True or FLAGS.loss_type == "L2":
    print("USING L2 DIFF")
    diffraction_loss = diffraction_loss_L2


  def pairCorr_loss_EM(pairCorr_fit, pairCorr_truth):
    return earth_movers_distance(pairCorr_fit, pairCorr_truth)

  def pairCorr_loss_L2(pairCorr_fit, pairCorr_truth):
    return L2_loss(pairCorr_fit, pairCorr_truth)

  def pairCorr_loss_WD(pairCorr_fit, pairCorr_truth):
    return wasserstein_distance(pairCorr_fit, pairCorr_truth, get_R(parameters))

  if FLAGS.loss_type == "EM":
    pairCorr_loss = pairCorr_loss_EM
  elif FLAGS.loss_type == "L2":
    pairCorr_loss = pairCorr_loss_L2
  elif FLAGS.loss_type == "WD":
    pairCorr_loss = pairCorr_loss_WD


  def momentum_loss(delta_geo):
    #dists = dist_metric_td(init_geo + delta_geo)
    #velocity = (dists[1:] - dists[:-1])/parameters["dt"]
    #return np.sum(np.square(velocity/FLAGS.momentum_scale))

    velocity2 = np.sum(np.square(delta_geo[1:,:,:] - delta_geo[:-1,:,:]))
    velocity2 /= parameters["dt"]**2
    momentum2 = velocity2*np.expand_dims(
        np.expand_dims(np.square(parameters["atom_masses"]), 0),
        0)

    momentum2 /= FLAGS.momentum_scale**2

    return np.mean(momentum2)


  def init_zero_loss(delta_geo):
    return np.sum(np.square(delta_geo[0,:,:]))

  return diffraction_loss, pairCorr_loss, momentum_loss, init_zero_loss


def get_calculate_losses_fxn(FLAGS, parameters, init_geo, q, r, scat_weights):
  
  diffraction_loss, pairCorr_loss,\
  momentum_loss, init_zero_loss = get_dependent_loss_fxns(
      FLAGS, parameters, init_geo)

  def calculate_losses(params, **kwargs):
      
    #print(kwargs)
    diffraction_truth, pairCorr_truth = kwargs["truth"]
    delta_geo = params
      
    diffraction_fit, pairCorr_fit = simulate_exp(
        init_geo, delta_geo, q, r, scat_weights)

    diffraction_loss_value  = diffraction_loss(
        diffraction_fit, diffraction_truth)
    pairCorr_loss_value     = pairCorr_loss(
        pairCorr_fit, pairCorr_truth)
    momentum_loss_value     = momentum_loss(delta_geo)
    init_zero_loss_value    = init_zero_loss(delta_geo)

    #print("CALCED LOSS", diffraction_loss_value, pairCorr_loss_value, momentum_loss_value, init_zero_loss_value)

    return diffraction_loss_value,\
        pairCorr_loss_value,\
        momentum_loss_value,\
        init_zero_loss_value

  return calculate_losses


def get_loss_fxn(FLAGS, parameters, init_geo, q, r, scat_weights):


  calculate_losses = get_calculate_losses_fxn(
      FLAGS, parameters, init_geo, q, r, scat_weights)


  def loss(params, **kwargs):
    
    loss_diffraction, loss_pairCorr, loss_momentum, loss_init_zero =\
        calculate_losses(params, **kwargs)

    if FLAGS.debugging:
      print("LOSS VAL", FLAGS.scale_diffraction_loss*loss_diffraction\
        + FLAGS.scale_pairCorr_loss*loss_pairCorr\
        + FLAGS.scale_init_zero_loss*loss_init_zero\
        + FLAGS.scale_momentum_loss*loss_momentum)
      print("LOSS VAL", FLAGS.scale_diffraction_loss*loss_diffraction,
        FLAGS.scale_pairCorr_loss*loss_pairCorr,
        FLAGS.scale_init_zero_loss*loss_init_zero,
        FLAGS.scale_momentum_loss*loss_momentum)
    return FLAGS.scale_diffraction_loss*loss_diffraction\
        + FLAGS.scale_pairCorr_loss*loss_pairCorr\
        + FLAGS.scale_init_zero_loss*loss_init_zero\
        + FLAGS.scale_momentum_loss*loss_momentum
  
  if FLAGS.debugging:
    return loss, calculate_losses
  else:
    return jax.jit(loss), jax.jit(calculate_losses)


#####  Save Fitting History  #####
def save_training_history(
    FLAGS, parameters, step,
    step_history,
    loss_history,
    dists_history,
    angle_history,
    opt_state,
    logging,
    grad_mag_history=None):


  save_dir      = os.path.join(parameters["history_dir"], FLAGS.molecule)
  flags_string  = str(FLAGS.flags_into_string())
  flags_hash    = blake2b(flags_string.encode('utf-8'), digest_size=10).hexdigest()
  prefix        = "flags-{}_step-{}_".format(flags_hash, step)
  logging.info("Saving history for hash " + flags_hash)

  flags_fileName = "flags-{}.txt".format(flags_hash)
  if not os.path.exists(os.path.join(save_dir, flags_fileName)):
    with open(os.path.join(save_dir, flags_fileName), "w") as file:
      file.write(flags_string)
  else:
    search = "flags-{}".format(flags_hash)
    prev_files = glob.glob(os.path.join(save_dir, search+"*.npy"))
    if len(prev_files) == 0:
      logging.fatal("Found {} but no other files".format(flags_fileName))
  
    prev_file = prev_files[0]
    sInd = prev_file.find("step-") + 5
    fInd = prev_file.find("_", sInd)
    time = int(prev_file[sInd:fInd])
    
    if time < step:
      # Remove older files
      for fl in prev_files:
        os.remove(fl)
      os.remove(os.path.join(save_dir, "flags-{}_step-{}_{}".format(
          flags_hash, time, "optimizer_state.pl")))
    elif time == step:
      logging.info("File at the same time already exists.")
    else:
      logging.fatal("Trying to save old results that exist ({}), should import"\
          .format(flags_hash))


  with open(os.path.join(save_dir, prefix+"step_history.npy"), "wb") as file:
    np.save(file, np.array(step_history))

  with open(os.path.join(save_dir, prefix+"loss_history.npy"), "wb") as file:
    np.save(file, np.concatenate(loss_history))

  with open(os.path.join(save_dir, prefix+"dists_history.npy"), "wb") as file:
    np.save(file, np.concatenate(dists_history))

  #with open(os.path.join(save_dir, prefix+"angle_history.npy"), "wb") as file:
  #  np.save(file, np.concatenate(angle_history))
 
  with open(os.path.join(save_dir, prefix+"optimizer_state.pl"), "wb") as file:
    pl.dump(optimizers.unpack_optimizer_state(opt_state), file)

  if grad_mag_history is not None:
    with open(os.path.join(save_dir, prefix+"gradient_mags_history.npy"), "wb") as file:
      np.save(file, np.array(grad_mag_history))



#####  Retrieving Fitting History  #####
def setup_fitting(
    FLAGS,
    parameters,
    opt_init,
    Npoints,
    geo_shape,
    logging):


  save_dir      = os.path.join(parameters["history_dir"], FLAGS.molecule)
  flags_string  = str(FLAGS.flags_into_string())
  flags_hash    = blake2b(flags_string.encode('utf-8'), digest_size=10).hexdigest()

  flags_fileName = "flags-{}.txt".format(flags_hash)
  if not os.path.exists(os.path.join(save_dir, flags_fileName)):

    #if FLAGS.debugging_opt is not None\
    #    and os.path.exists(os.path.join(save_dir, flags_fileName)):
    #  os.remove(os.path.join(save_dir, "flags-{}*".format(flags_hash)))

    logging.info("Cannot find history for {} start from init.".format(flags_hash))
    delta_geo = np.zeros((Npoints, geo_shape[0], geo_shape[1]))
    opt_state = opt_init(delta_geo)
    return 0, [], [], [], [], opt_state
  else:
    search = "flags-{}".format(flags_hash)
    prev_files = glob.glob(os.path.join(save_dir, search+"*.npy"))
    if len(prev_files) == 0:
      logging.fatal("Found {} but no other files to import".format(
          flags_fileName))
  
    prev_file = prev_files[0]
    sInd = prev_file.find("step-") + 5
    fInd = prev_file.find("_", sInd)
    step = int(prev_file[sInd:fInd])
    prefix = "flags-{}_step-{}_".format(flags_hash, step)

   
    print(os.path.join(save_dir, prefix+"step_history.npy"))
    with open(os.path.join(save_dir, prefix+"step_history.npy"), "rb") as file:
      step_history  = np.load(file).tolist()

    with open(os.path.join(save_dir, prefix+"loss_history.npy"), "rb") as file:
      loss_history  = [np.load(file)]

    with open(os.path.join(save_dir, prefix+"dists_history.npy"), "rb") as file:
      dists_history = [np.load(file)]

    #with open(os.path.join(save_dir, prefix+"angle_history.npy"), "rb") as file:
    #  angle_history = [np.load(file)]
   
    with open(os.path.join(save_dir, prefix+"optimizer_state.pl"), "rb") as file:
      opt_state = optimizers.pack_optimizer_state(pl.load(file))

    return step+1, step_history, loss_history, dists_history, [], opt_state



def main(argv):
  logging.info("Running main")

  rng = random.PRNGKey(0)
  parameters = get_parameters()

  step_history  = []
  loss_history  = []
  dists_history = []
  angle_history = []

  ######################
  #####  Get Data  #####
  ######################

  init_geo, atom_positions,\
  dists_data, angles_data,\
  mmd_data, pairCorr_data = get_data(FLAGS, parameters)
  
  q = get_Q(parameters)
  r = get_R(parameters)
  scat_amps, scat_weights = get_scattering_amps(parameters, q)


  ###############################
  #####  Get Loss Function  #####
  ###############################

  loss, calculate_losses = get_loss_fxn(
      FLAGS, parameters, init_geo, q, r, scat_weights)


  #############################
  #####  Setup Optimizer  #####
  #############################

  if FLAGS.optimizer == "SGD":
    opt_init, opt_update, get_params = optimizers.sgd(FLAGS.SGD_LR)
  elif FLAGS.optimizer == "ADAM":
    opt_init, opt_update, get_params = optimizers.adam(1e-4)
  elif FLAGS.optimizer == "FIRE":
    opt_init_, opt_update = minimize.fire_descent(loss, shift)
    def opt_init(R, **kwargs):
      return opt_init_(R + init_geo, **kwargs)
    def get_params(opt_state):
      R, V, F, ds, alpha, f = opt_state
      return R - init_geo
  else:
    logging.fatal("Do not recognize opimizer{}".format(FLAGS.optimizer))


  ###############################
  #####  Setup Environment  #####
  ###############################

  print("SETTING UP ENV")
  step, step_history,\
  loss_history,\
  dists_history,\
  angle_history,\
  opt_state = setup_fitting(
      FLAGS, parameters, opt_init,
      dists_data.shape[0],
      (init_geo.shape[-2], init_geo.shape[-1]),
      logging)

  calc_grad_mags = False
  if FLAGS.debugging_opt is not None:
    if "init_to_data" in FLAGS.debugging_opt:
      opt_state = opt_init(atom_positions - init_geo)
    if "init_to_noisy_data" in FLAGS.debugging_opt:
      init = atom_positions - init_geo
      rng, k = random.split(rng)
      opt_state = opt_init(init + random.normal(k, init.shape)*0.1)
    if "calc_grad_mag" in FLAGS.debugging_opt:
      calc_grad_mags = True
      grad_mag_history = []

  
  gradient = grad(loss)
  def update(stp, opt_state, **kwargs):

    params = get_params(opt_state)
  
    if FLAGS.debugging:
      gg = gradient(params, **kwargs)
      print("GRADAS", gg[0])
      return opt_update(stp, gg, opt_state)
    else:
      return opt_update(stp, gradient(params, **kwargs), opt_state)
  
  if not FLAGS.debugging:
    update = jit(update)


  def get_grad_mags(opt_state, history):
    
    params = get_params(opt_state)
    history.append(np.sqrt(np.sum(np.square(params)))/params.shape[0])

    return history


  def asses_fit(opt_state, **kwargs):
    params = get_params(opt_state)

    loss_diffraction, loss_pairCorr, loss_momentum, loss_init_zero =\
        calculate_losses(params, **kwargs)

    loss_diffraction  *= FLAGS.scale_diffraction_loss
    loss_pairCorr     *= FLAGS.scale_pairCorr_loss
    loss_momentum     *= FLAGS.scale_momentum_loss
    loss_init_zero    *= FLAGS.scale_init_zero_loss
    losses = [loss_diffraction, loss_pairCorr, loss_momentum, loss_init_zero]

    return np.sum(losses), losses
  
  if not FLAGS.debugging:
    asses_fit = jit(asses_fit)



  def append_history(stp, losses, delta_geo, init_geo):
    dists   = dist_metric_td(init_geo + delta_geo)
    angles  = angle_metric_td(init_geo + delta_geo)

    step_history.append(stp)
    loss_history.append(np.expand_dims(np.array(losses), 0))
    dists_history.append(np.expand_dims(dists, 0))
    #angle_history.append(np.expand_dims(angles, 0))

 
  data = (mmd_data, pairCorr_data)

  # Evaluate initial conditions
  params = get_params(opt_state)
  loss_val, losses = asses_fit(opt_state, truth=data)

  append_history(step, losses, params, init_geo)

  log_info = "Step {}\tLoss {} / {}".format(step, loss_val, losses)
  logging.info(log_info)

  end_step = step + FLAGS.Nsteps
  while step < end_step or FLAGS.Nsteps < 0:
    #print("STARTING LOOP")
    params = get_params(opt_state)

    #print("UPDATING")
    opt_state = update(step, opt_state, truth=data)
    step += 1
    #print("FINISHED UPDATE")

    if step % parameters["log_every"] == 0:
      loss_val, losses, = asses_fit(opt_state, truth=data)

      append_history(step, losses, params, init_geo)
      if calc_grad_mags:
        grad_mag_history = get_grad_mags(opt_state, grad_mag_history)

      log_info = "Step {}\tLoss {} / {}".format(step, loss_val, losses)
      logging.info(log_info)

    if step % parameters["save_every"] == 0:
      logging.info("Saving at step {}".format(step))
      save_training_history(
          FLAGS, parameters, step,
          step_history,
          loss_history,
          dists_history,
          angle_history,
          opt_state,
          logging,
          grad_mag_history=grad_mag_history)


  save_training_history(
      FLAGS, parameters,
      FLAGS.Nsteps,
      step_history,
      loss_history,
      dists_history,
      angle_history,
      opt_state,
      logging,
      grad_mag_history=grad_mag_history)


if __name__ == "__main__":

  app.run(main)
