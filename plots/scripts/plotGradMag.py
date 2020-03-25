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
from modules.flags.flags_WD import get_flags
from modules.diffraction import *


FLAGS = get_flags()


def make_plot(
    q, r, losses,
    step_history,
    loss_history, grad_mags_history,
    colors, atoms, plot_dir):


  fig, axs = plt.subplots(2, 1, figsize=(7,10))

  #########################
  #####  Plot Losses  #####
  #########################

  for i,lss in enumerate(losses):
    axs[0].plot(step_history, loss_history[:,i], label=lss)
  axs[0].plot(step_history, onp.sum(loss_history, axis=1), 
      '-k', label="Total Loss")
  axs[0].set_xlim(step_history[0], step_history[-1])
  axs[0].set_yscale('log')
  axs[0].legend()

  ######################################
  #####  Plot Gradient Magnitudes  #####
  ######################################

  axs[1].plot(step_history[:-1], grad_mags_history)
  axs[1].set_xlim(step_history[0], step_history[-1])
  axs[1].set_xlabel("Time [fs]")
  axs[1].set_ylabel(r"Gradient Magnitudes [$\AA$]")
  axs[1].set_yscale('log')
  axs[1].legend()

  #fig.tight_layout()
  fig.savefig(
      os.path.join(plot_dir, "gradient_magnitude.png"))
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

  with open(os.path.join(hist_dir, prefix+"gradient_mags_history.npy"), "rb") as file:
    grad_mags_history = np.load(file)



  ######################
  #####  Plotting  #####
  ######################

  losses  = [
      "Diffraction Loss", "Pair Correlation Loss",
      "Momentum Loss", "Initial Conditions Loss"]
  colors  = ['r', 'b', 'g']
  atoms   = ['C', 'H', 'O']

  # Check if directory exists
  plot_dir = os.path.join("../", FLAGS.molecule, flags_hash)
  if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

  #####  Run with Multiprocessing  #####
  make_plot( 
    q, r,
    losses,
    step_history,
    loss_history,
    grad_mags_history,
    colors,
    atoms,
    plot_dir)

  print("Finished with flags {}".format(flags_hash))


if __name__ == "__main__":

  app.run(main)

