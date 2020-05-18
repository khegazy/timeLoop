from absl import app
from absl import logging

import jax as jax
import jax.lax as lax
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

  if True and FLAGS.loss_type == "EM":
    diffraction_loss = diffraction_loss_EM
  elif True or FLAGS.loss_type == "L2":
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


  def kinematic_loss(delta_geo):
    #dists = dist_metric_td(init_geo + delta_geo)
    #velocity = (dists[1:] - dists[:-1])/parameters["dt"]
    #return np.sum(np.square(velocity/FLAGS.momentum_scale))

    """
    velocity2 = np.sum(np.square(delta_geo[1:,:,:] - delta_geo[:-1,:,:]))
    velocity2 /= parameters["dt"]**2
    momentum2 = velocity2*np.expand_dims(
        np.expand_dims(np.square(parameters["atom_masses"]), 0),
        0)

    momentum2 /= FLAGS.momentum_scale**2
    """

    _, momentum, _, force, expect_pos = calc_kinematics(
        delta_geo, parameters["atom_masses"], parameters["dt"])

    momentum_loss = np.sum(np.square(momentum), axis=-1)
    momentum_loss = np.mean(momentum_loss, axis=-1)
    momentum_loss /= FLAGS.momentum_scale**2

    force_loss = np.sum(np.square(force), axis=-1)
    force_loss = np.mean(force_loss, axis=-1)
    force_loss /= FLAGS.force_scale**2

    expect_pos_loss = np.sum(np.square(delta_geo - expect_pos), axis=-1)
    expect_pos_loss = np.mean(expect_pos_loss, axis=-1)

    return momentum_loss, force_loss, expect_pos_loss


  def init_zero_loss(delta_geo):
    return np.sum(np.square(delta_geo[0,:,:]))

  return diffraction_loss, pairCorr_loss, kinematic_loss, init_zero_loss


def get_calculate_losses_fxn(FLAGS, parameters, init_geo, q, r, scat_weights):
  
  diffraction_loss, pairCorr_loss,\
  kinematic_loss, init_zero_loss = get_dependent_loss_fxns(
      FLAGS, parameters, init_geo)
  labels = []
  scales = []

  def calculate_losses(params, **kwargs):
      
    diffraction_truth, pairCorr_truth = kwargs["truth"]
    #weight_tm = kwargs["weight_tm"]
    delta_geo = params
    print("delta geo", delta_geo)
    
    diffraction_fit, pairCorr_fit = simulate_exp(
        init_geo, delta_geo, q, r, scat_weights)

    diffraction_loss_value  = np.mean(diffraction_loss(
                                diffraction_fit, diffraction_truth))
    pairCorr_loss_value     = np.mean(pairCorr_loss(
                                pairCorr_fit, pairCorr_truth))
    init_zero_loss_value    = init_zero_loss(delta_geo)
    momentum_loss_value,\
    force_loss_value,\
    expect_pos_loss_value   = kinematic_loss(delta_geo)
    momentum_loss_value     = np.mean(momentum_loss_value)
    force_loss_value        = np.mean(force_loss_value)
    expect_pos_loss_value   = np.mean(expect_pos_loss_value)


    #print("CALCED LOSS", diffraction_loss_value, pairCorr_loss_value, momentum_loss_value, init_zero_loss_value)

    return np.array([
        diffraction_loss_value,\
        pairCorr_loss_value,\
        init_zero_loss_value,\
        momentum_loss_value,\
        force_loss_value,
        expect_pos_loss_value])

  labels = [
      "Diffraction Loss", "Pair Corr Loss",\
      "Init Zero Loss", "Momentum Loss",\
      "Force Loss", "Expected Position Loss"]
  scales = np.array([
      FLAGS.scale_diffraction_loss, FLAGS.scale_pairCorr_loss,
      FLAGS.scale_init_zero_loss, FLAGS.scale_momentum_loss,
      FLAGS.scale_force_loss, FLAGS.scale_expect_pos_loss])




  return labels, scales, calculate_losses


def get_loss_fxn(FLAGS, parameters, init_geo, q, r, scat_weights):


  labels, scales, calculate_losses = get_calculate_losses_fxn(
      FLAGS, parameters, init_geo, q, r, scat_weights)


  def loss(params, **kwargs):
   
    print("ENTERED LOSS")
    #loss_diffraction, loss_pairCorr,\
    #loss_init_zero, loss_momentum, loss_force =\
    losses = calculate_losses(params, **kwargs)

    if FLAGS.debugging:
      print("LOSS VAL")
      print(loss_diffraction)
      print(loss_pairCorr)
      print(loss_init_zero)
      print(loss_momentum)

      print("LOSS VAL", FLAGS.scale_diffraction_loss*loss_diffraction\
        + FLAGS.scale_pairCorr_loss*loss_pairCorr\
        + FLAGS.scale_init_zero_loss*loss_init_zero\
        + FLAGS.scale_momentum_loss*loss_momentum)
      print("LOSS VAL", FLAGS.scale_diffraction_loss*loss_diffraction,
        FLAGS.scale_pairCorr_loss*loss_pairCorr,
        FLAGS.scale_init_zero_loss*loss_init_zero,
        FLAGS.scale_momentum_loss*loss_momentum)
    return np.sum(scales*losses)
    """
    FLAGS.scale_diffraction_loss*loss_diffraction\
        + FLAGS.scale_pairCorr_loss*loss_pairCorr\
        + FLAGS.scale_init_zero_loss*loss_init_zero\
        + FLAGS.scale_momentum_loss*loss_momentum\
        + FLAGS.scale_force_loss*loss_force
    """
  
  if FLAGS.debugging:
    return loss, calculate_losses, labels, scales
  else:
    return jax.jit(loss), jax.jit(calculate_losses), labels, scales


#####  Save Fitting History  #####
def save_training_history(
    FLAGS, parameters, step,
    step_history,
    loss_history,
    dists_history,
    angle_history,
    geo_history,
    opt_state,
    logging,
    grad_mag_history=None):


  flags_string  = str(FLAGS.flags_into_string())
  flags_hash    = blake2b(flags_string.encode('utf-8'), digest_size=10).hexdigest()
  prefix        = "flags-{}_step-{}_".format(flags_hash, step)
  save_dir      = os.path.join(parameters["history_dir"], FLAGS.molecule)
  if FLAGS.experiment is not None:
    save_dir = os.path.join(save_dir, FLAGS.experiment)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
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
 
  with open(os.path.join(save_dir, prefix+"geometry_history.npy"), "wb") as file:
    np.save(file, np.concatenate(geo_history))

  with open(os.path.join(save_dir, prefix+"optimizer_state.pl"), "wb") as file:
    pl.dump(optimizers.unpack_optimizer_state(opt_state), file)

  if grad_mag_history is not None:
    with open(os.path.join(save_dir, prefix+"gradient_mags_history.npy"), "wb") as file:
      np.save(file, np.array(grad_mag_history))



#####  Retrieving Fitting History  #####
def setup_fitting(
    FLAGS,
    parameters,
    data,
    times,
    opt_init,
    Npoints,
    geo_shape,
    data_schedule,
    logging):



  flags_string  = str(FLAGS.flags_into_string())
  flags_hash    = blake2b(flags_string.encode('utf-8'), digest_size=10).hexdigest()
  save_dir      = os.path.join(parameters["history_dir"], FLAGS.molecule)
  if FLAGS.experiment is not None:
    save_dir = os.path.join(save_dir, FLAGS.experiment)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

  flags_fileName = "flags-{}.txt".format(flags_hash)
  if not os.path.exists(os.path.join(save_dir, flags_fileName)):

    #if FLAGS.debugging_opt is not None\
    #    and os.path.exists(os.path.join(save_dir, flags_fileName)):
    #  os.remove(os.path.join(save_dir, "flags-{}*".format(flags_hash)))

    logging.info("Cannot find history for {} start from init.".format(flags_hash))
   
    data_sched_step = None
    if data_schedule is not None:
      data_sched_step, data_sched_time = data_schedule.pop(0)
      print(data_sched_step, data_sched_time)
      data_cut = np.argmin(np.abs(data_sched_time - times)) + 1
      Npoints = data_cut

      data_fit = []
      for dt in data:
        data_fit.append(dt[:data_cut])
      data = tuple(data_fit)
    
    delta_geo = np.zeros((Npoints, geo_shape[0], geo_shape[1]))
    opt_state = opt_init(delta_geo)

    return data, 0, [], [], [], [], [], data_schedule, data_sched_step, opt_state
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

    data_sched_step = None
    if data_schedule is not None:
      data_sched_step = -1
      while step > data_sched_step:
        data_sched_step, data_sched_time = data_schedule.pop(0)
      data_cut = np.argmin(np.abs(data_sched_time - times)) + 1
        
      data_fit = []
      for dt in data:
        data_fit.append(dt[:data_cut])
      data = tuple(data_fit)
   
    print(os.path.join(save_dir, prefix+"step_history.npy"))
    with open(os.path.join(save_dir, prefix+"step_history.npy"), "rb") as file:
      step_history  = np.load(file).tolist()

    with open(os.path.join(save_dir, prefix+"loss_history.npy"), "rb") as file:
      loss_history  = [np.load(file)]

    with open(os.path.join(save_dir, prefix+"dists_history.npy"), "rb") as file:
      dists_history = [np.load(file)]

    #with open(os.path.join(save_dir, prefix+"angle_history.npy"), "rb") as file:
    #  angle_history = [np.load(file)]
    
    with open(os.path.join(save_dir, prefix+"geometry_history.npy"), "rb") as file:
      geo_history = [np.load(file)]
   
    with open(os.path.join(save_dir, prefix+"optimizer_state.pl"), "rb") as file:
      opt_state = optimizers.pack_optimizer_state(pl.load(file))

    return data,\
        step+1, step_history,\
        loss_history, dists_history,\
        [], geo_history,\
        data_schedule, data_sched_step,\
        opt_state



def main(argv):
  logging.info("Running main")

  rng = random.PRNGKey(0)
  parameters = get_parameters()

  step_history  = []
  loss_history  = []
  dists_history = []
  angle_history = []
  grad_mag_history = []

  ######################
  #####  Get Data  #####
  ######################

  init_geo, atom_positions,\
  dists_data, angles_data,\
  mmd_data, pairCorr_data = get_data(FLAGS, parameters)
  data = (mmd_data, pairCorr_data)
  Ndata = mmd_data.shape[0]
  times = np.arange(Ndata)*parameters["dt"]
  
  q = get_Q(parameters)
  r = get_R(parameters)
  scat_amps, scat_weights = get_scattering_amps(parameters, q)


  ###############################
  #####  Get Loss Function  #####
  ###############################

  loss, calculate_losses, loss_labels, loss_scales = get_loss_fxn(
      FLAGS, parameters, init_geo, q, r, scat_weights)


  #############################
  #####  Setup Optimizer  #####
  #############################

  data_schedule = [(np.inf, 1), (60, 6), (90, 9), (120, 12),
      (150, 15), (130, 30), (np.inf, 100)]
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
  print(init_geo.shape, dists_data.shape)
  data_fit,\
  step, step_history,\
  loss_history,\
  dists_history,\
  angle_history,\
  geo_history,\
  data_schedule,\
  data_sched_step,\
  opt_state = setup_fitting(
      FLAGS, parameters,
      data, times, opt_init,
      dists_data.shape[0],
      (init_geo.shape[-2], init_geo.shape[-1]),
      data_schedule,
      logging)

  calc_grad_mags = True
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
 
    grad = gradient(params, **kwargs)
    #print("GRAD")
    #print(grad)

    """
    Ntime = grad.shape[0]
    w_cut = np.argmin(kwargs["weight_tm"])
    cp = np.abs(((Ntime-w_cut+1) % (Ntime+1)) - 1).astype(int)
    #cp = np.max([Ntime-w_cut % Ntime, 1]).astype(int)
    fill = np.tile(grad[w_cut-1,:,:],
        (cp, 1, 1))
    print("w",w_cut, cp, fill.shape)
    insert = np.abs((np.abs(w_cut-(Ntime-1)) % (Ntime-1)) - (Ntime-1))
    #print("w_cut", w_cut, insert, onp.array(kwargs["weight_tm"])[w_cut])
    #print("tiled", fill.shape)
    grad = lax.lax.dynamic_update_slice(grad,
      fill,
      [insert,0,0])
    #w_cut = w_cut - grad.shape[0] % grad.shape[0]
    #grad = np.concatenate([grad[w_cut:,:,:],
    #  np.tile(grad[w_cut-1,:,:], (np.abs(w_cut), 1, 1))])
    """

    return opt_update(stp, grad, opt_state)
  
  if not FLAGS.debugging:
    update = jit(update)

  
  def update_unweighted_params(opt_state, weight_cut):

    params = get_params(opt_state)
    op = onp.array(params)
    if weight_cut <= op.shape[0]:
      op[weight_cut:] = op[weight_cut]
    params = np.array(op)
    """
    params_.append(np.array(op))
        np.concat(
        [p[:weight_cut],
          np.tile(p[weight_cut], (p.shape[0] - weight_cut, 1))],
        axis=0))
    """

    return (params,) + opt_state[0:]


  def get_grad_mags(opt_state, history):
    
    params = get_params(opt_state)
    history.append(np.sqrt(np.sum(np.square(params)))/params.shape[0])

    return history


  def asses_fit(opt_state, **kwargs):
    print("1")
    params = get_params(opt_state)

    print(2)
    losses = calculate_losses(params, **kwargs)

    print(3)


    losses = losses*kwargs["loss_scales"]
    #losses = [loss_diffraction, loss_pairCorr, loss_init_zero, loss_momentum, loss_force]
    print("4")
    print(losses)
    print(np.sum(losses))
    print("DONE")
    return np.sum(losses), losses
  
  if not FLAGS.debugging:
    asses_fit = jit(asses_fit)



  def append_history(stp, losses, opt_state, init_geo, Ntimes=None):
    delta_geo  = get_params(opt_state)
    if Ntimes is not None:
      if delta_geo.shape[0] < Ntimes:
        delta_geo = np.concatenate([delta_geo,
            np.tile(np.expand_dims(delta_geo[-1,:,:], axis=0),
              (Ntimes-delta_geo.shape[0],1,1))],
            axis=0)

    dists   = dist_metric_td(init_geo + delta_geo)
    angles  = angle_metric_td(init_geo + delta_geo)

    step_history.append(stp)
    loss_history.append(np.expand_dims(losses, 0))
    dists_history.append(np.expand_dims(dists, 0))
    #angle_history.append(np.expand_dims(angles, 0))
    geo_history.append(np.expand_dims(delta_geo + init_geo, 0))

 
  data = (mmd_data, pairCorr_data)

  # Evaluate initial conditions
  weight_tm = np.ones(Ndata)
  #weight_std = 20
  #w_arg = np.arange(6*weight_std+1) - 3*weight_std
  while step > data_sched_step:
    data_sched_step, data_sched_time = data_schedule.pop(0)
    data_cut = np.argmin(np.abs(data_sched_time - times)) + 1

    # Update the range of the data
    data_fit = []
    for dt in data:
      data_fit.append(dt[:data_cut])
    data_fit = tuple(data_fit)


  print("ASSESING", len(data_fit))
  print(asses_fit(opt_state,
      truth=data_fit, loss_scales=loss_scales))
  loss_val, losses = asses_fit(opt_state,
      truth=data_fit, loss_scales=loss_scales)

  print("apending")
  append_history(step, losses, opt_state, init_geo, Ntimes=times.shape[0])

  print("printing")
  log_info = "Step {}\tLoss {} / {}".format(step, loss_val, losses)
  logging.info(log_info)

  end_step = step + FLAGS.Nsteps
  while step < end_step or FLAGS.Nsteps < 0:
    #print("STARTING LOOP")

    if data_sched_step is not None:
      if step > data_sched_step:
        if len(data_schedule) == 0:
          logging.fatal("Data schedule needs a final schedule.")
        data_sched_step, data_sched_time = data_schedule.pop(0)
        data_cut = np.argmin(np.abs(data_sched_time - times)) + 1
        
        # Update the range of the data
        data_fit = []
        for dt in data:
          data_fit.append(dt[:data_cut])
        data_fit = tuple(data_fit)

        delta_geo = get_params(opt_state)
        opt_state = opt_init(
            np.concatenate([delta_geo,
              np.tile(delta_geo[-1,:,:], (data_cut-delta_geo.shape[0],1,1))],
              axis=0))
        delta_geo = get_params(opt_state)

      """
      w_ind += 1
      weight_step, w_time = weight_schedule[w_ind]
      weight_cut = np.argmin(np.abs(times - w_time)) + 1
      weight_tm = np.concatenate(
          [np.ones(weight_cut), np.zeros(Ndata-weight_cut)])
      """

    """
    if weight_step > 0:
      weight_step = 0
    weight_tm = onp.zeros(Ndata)
    sInd = np.max([0, weight_step-3*weight_std])
    eInd = np.min([weight_step+3*weight_std, Ndata])
    #print("SHAPESSSSSSSSSSSSSS", sInd, eInd, weight_tm[sInd:eInd].shape, np.exp(-0.5*(w_arg[:eInd-sInd]/weight_std)**2).shape)
    if weight_step <= 3*weight_std:
      weight_tm[sInd:eInd] = np.exp(-0.5*(w_arg[-1*(eInd-sInd):]/weight_std)**2)/\
          (weight_std*np.sqrt(2*np.pi))
    else:
      weight_tm[sInd:eInd] = np.exp(-0.5*(w_arg[:eInd-sInd]/weight_std)**2)/\
          (weight_std*np.sqrt(2*np.pi))
    weight_tm = np.array(weight_tm)
    """

    #print(opt_state)
    #print("elkajdflkajsdf")
    #print(optimizers.unpack_optimizer_state(opt_state))
    #print("UPDATING")
    opt_state = update(step, opt_state,
        truth=data_fit, weight_tm=weight_tm)
    #opt_state = update_unweighted_params(opt_state, weight_cut)
    #print(opt_state)
    step += 1
    #print("FINISHED UPDATE")

    if step % parameters["log_every"] == 0:
      print("APPENDING", step)
      loss_val, losses, = asses_fit(opt_state,
          truth=data_fit, loss_scales=loss_scales)

      append_history(step, losses, opt_state, init_geo, Ntimes=times.shape[0])
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
          geo_history,
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
      geo_history,
      opt_state,
      logging,
      grad_mag_history=grad_mag_history)


if __name__ == "__main__":

  app.run(main)
