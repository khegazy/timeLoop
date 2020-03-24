from absl import app
from absl import logging

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from hashlib import blake2b
from multiprocessing import Pool

from modules.parameters import get_parameters
from modules.flags.flags_EM import get_flags
from modules.diffraction import *


FLAGS = get_flags()

def simulate_exp(dists, q, r, scat_weights):

  mm_diffraction, _, _ = diffraction(
      dists, q, scat_weights)
  pair_corr = calc_pair_corr(q, r, mm_diffraction, 3.5)

  return mm_diffraction, pair_corr


def make_plot(istp,
    q, r, scat_weights, losses,
    sim_times, dfpc_times, dfpc_inds,
    step_history, loss_history,
    dists_history, dists_truth,
    diffraction_truth, pair_corr_truth,
    diffraction_hist, pair_corr_hist,
    timeCut, colors, atoms, plot_dir):


  stp = step_history[istp]
  print("Plotting time {}".format(istp))

  fig, axs = plt.subplots(2,1+len(dfpc_times), figsize=(21,10))

  #########################
  #####  Plot Losses  #####
  #########################

  for i,lss in enumerate(losses):
    axs[0,0].plot(step_history[:istp], loss_history[:istp,i], label=lss)
  axs[0,0].plot(step_history[:istp], onp.sum(loss_history[:istp,:], axis=1), 
      '-k', label="Total Loss")
  axs[0,0].set_xlim(step_history[0], step_history[istp])
  axs[0,0].set_yscale('log')
  axs[0,0].legend()

  #####################################
  #####  Plot Pairwise Distances  #####
  #####################################

  ci = 0
  for i in range(3):
    for j in range(i+1, 3):
      axs[1,0].plot(
          sim_times,
          dists_truth[:timeCut,i,j],
          c=colors[ci],
          label="{} - {}".format(atoms[i], atoms[j]))
      axs[1,0].plot(
          sim_times,
          dists_history[istp,:timeCut,i,j],
          c=colors[ci], linestyle=":")
      ci += 1
  axs[1,0].set_xlim([sim_times[0], sim_times[-1]])
  axs[1,0].set_xlabel("Time [fs]")
  axs[1,0].set_ylabel(r"Pair Distance [$\AA$]")
  axs[1,0].legend()


  ###################################################
  #####  Plot Diffraction and Pair Correlation  #####
  ###################################################

  for i,tm in enumerate(dfpc_inds):
    axs[0,i+1].plot(q, diffraction_truth[i,:], '-k')
    axs[1,i+1].plot(r, pair_corr_truth[i,:], '-k')
    axs[0,i+1].plot(q, diffraction_hist[istp,i,:], '-b')
    axs[1,i+1].plot(r, pair_corr_hist[istp,i,:], '-b')

    axs[0,i+1].set_title("Fit at time {} fs".format(dfpc_times[i]))

  #fig.tight_layout()
  fig.savefig(
      os.path.join(plot_dir, "dataFit_{}.png".format(stp)))
  plt.close()


def main(argv):
  ##################################
  #####  Get History and Data  #####
  ##################################

  parameters = get_parameters()
  dt = 0.25
  data_dir = "../../data/" + FLAGS.molecule

  hist_dir      = os.path.join(parameters["history_dir"], FLAGS.molecule)
  flags_string  = str(FLAGS.flags_into_string())
  flags_hash    = blake2b(flags_string.encode('utf-8'), digest_size=10).hexdigest()
  print("HASH",flags_hash)

  q = get_Q(parameters)
  r = get_R(parameters)
  scat_amps, scat_weights = get_scattering_amps(parameters, q)

  search = "flags-{}".format(flags_hash)
  prev_files = glob.glob(
      os.path.join(parameters["history_dir"], FLAGS.molecule, search+"*.npy"))
  print("looking in ", os.path.join(parameters["history_dir"], FLAGS.molecule, search+"*.npy"))
  if len(prev_files) == 0:
    logging.fatal("Cannot find history files with hash {}".format(
        flags_hash))

  prev_file = prev_files[0]
  sInd = prev_file.find("step-") + 5
  fInd = prev_file.find("_", sInd)
  step = int(prev_file[sInd:fInd])
  prefix = "flags-{}_step-{}_".format(flags_hash, step)


  with open(os.path.join(hist_dir, prefix+"step_history.npy"), "rb") as file:
    step_history = np.load(file)

  with open(os.path.join(hist_dir, prefix+"loss_history.npy"), "rb") as file:
    loss_history = np.load(file)

  with open(os.path.join(hist_dir, prefix+"dists_history.npy"), "rb") as file:
    dists_history = np.load(file)

  with open(os.path.join(data_dir, "pairwise_distances.npy"), "rb") as file:
    dists_truth = np.load(file)



  ######################
  #####  Plotting  #####
  ######################

  losses  = [
      "Diffraction Loss", "Pair Correlation Loss",
      "Velocity Loss", "Initial Conditions Loss"]
  colors  = ['r', 'b', 'g']
  atoms   = ['C', 'H', 'O']
  timeCut = 400
  sim_times = np.arange(timeCut)*dt

  dfpc_times  = np.array([3., 6., 22.]) #fs
  dfpc_inds   = []
  for tm in dfpc_times:
    dfpc_inds.append(np.argmin(np.abs(sim_times - tm)))
  dfpc_inds = np.array(dfpc_inds)

  # Check if directory exists
  plot_dir = os.path.join("../", FLAGS.molecule, flags_hash)
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

  # Simulate experiment
  diffraction_truth, pair_corr_truth = simulate_exp(
      dists_truth[dfpc_inds,:], q, r, scat_weights)
  diffraction_hist  = onp.zeros(
      (len(step_history), len(dfpc_inds), diffraction_truth.shape[1]))
  pair_corr_hist    = onp.zeros(
      (len(step_history), len(dfpc_inds), pair_corr_truth.shape[1]))
  for istp in range(len(step_history)):
    diffraction_hist[istp,:,:], pair_corr_hist[istp,:,:] = simulate_exp(
        dists_history[istp,dfpc_inds,:], q, r, scat_weights)


  #####  Run with Multiprocessing  #####
  pool = Pool(processes=20)
  res = pool.map(partial(make_plot, 
    q=q, r=r,
    scat_weights=scat_weights,
    losses=losses,
    sim_times=sim_times,
    dfpc_times=dfpc_times,
    dfpc_inds=dfpc_inds,
    step_history=step_history,
    loss_history=loss_history,
    dists_history=dists_history,
    dists_truth=dists_truth,
    diffraction_truth=diffraction_truth,
    pair_corr_truth=pair_corr_truth,
    diffraction_hist=diffraction_hist,
    pair_corr_hist=pair_corr_hist,
    timeCut=timeCut,
    colors=colors,
    atoms=atoms,
    plot_dir=plot_dir),
    range(len(step_history)))

  print("Finished with flags {}".format(flags_hash))


if __name__ == "__main__":

  app.run(main)

