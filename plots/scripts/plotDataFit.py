from absl import app
from absl import logging

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
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


def orient_CHO(geo):
 
  geo -= geo[0,:]
  #delta = geo[1,:] - geo[-1,:]
  #geo[-1,0] = 0
  #geo[-1,1] = norm(geo[0,:]-geo[-1,:])
  #geo[-1,2] = 0
  #geo[1,:] = geo[-1,:] + delta

  return geo


def make_plot(istp,
    q, r, scat_weights, losses,
    sim_times, dfpc_times, dfpc_inds,
    step_history, loss_history,
    geo_history, geo_truth,
    dists_history, dists_truth,
    diffraction_truth, pair_corr_truth,
    diffraction_hist, pair_corr_hist,
    Ntime_steps, colors, atoms, plot_dir):


  stp = step_history[istp]
  if stp % 2 != 0:
    return

  fig = plt.figure(figsize=(21,10))
  grid1 = plt.GridSpec(2, 1 + len(dfpc_times),
      left=0.05, right=0.95,
      top=0.95, bottom=0.18,
      hspace=0.16, wspace=0.14)
  grid2 = plt.GridSpec(1, 3 + 3*len(dfpc_times),
      left=0.05, right=0.95,
      top=0.13, bottom=0.02,
      hspace=0.11, wspace=0.1)


  axLosses  = fig.add_subplot(grid1[0,0])
  axDists   = fig.add_subplot(grid1[1,0])
  axDiffs,axPCs = [], []

  for i in range(len(dfpc_times)):
    j=i+1
    axDiffs.append(fig.add_subplot(grid1[0,j]))
    axPCs.append(fig.add_subplot(grid1[1,j]))




  #########################
  #####  Plot Losses  #####
  #########################

  for i,lss in enumerate(losses):
    axLosses.plot(step_history[:istp], loss_history[:istp,i], label=lss)
  #axLosses.plot(step_history[:istp], onp.sum(loss_history[:istp,:], axis=1), 
  #    '-k', label="Total Loss")
  axLosses.set_xlim(step_history[0], step_history[istp])
  axLosses.set_xlabel("Training Time [steps]")
  #axs[0,0].set_xscale('log')
  axLosses.set_yscale('log')
  axLosses.set_ylabel("Loss")
  axLosses.legend()

  #####################################
  #####  Plot Pairwise Distances  #####
  #####################################

  ci = 0
  for i in range(3):
    for j in range(i+1, 3):
      axDists.plot(
          sim_times,
          dists_truth[:Ntime_steps,i,j],
          c=colors[ci],
          label="{} - {}".format(atoms[i], atoms[j]))
      axDists.plot(
          sim_times,
          dists_history[istp,:Ntime_steps,i,j],
          c=colors[ci], linestyle=":")
      ci += 1
  axDists.set_xlim([sim_times[0], sim_times[-1]])
  axDists.set_xlabel("Time [fs]")
  axDists.set_ylabel(r"Pair Distance [$\AA$]")
  axDists.legend()


  ###################################################
  #####  Plot Diffraction and Pair Correlation  #####
  ###################################################

  for i,tm in enumerate(dfpc_inds):
    axDiffs[i].plot(q, diffraction_truth[i,:], '-k')
    axDiffs[i].plot(q, diffraction_hist[istp,i,:], '-b')
    axDiffs[i].set_xlabel(r"Q [$\AA^{-1}$]")
    axPCs[i].plot(r, pair_corr_truth[i,:], '-k')
    axPCs[i].plot(r, pair_corr_hist[istp,i,:], '-b')
    axPCs[i].set_xlabel(r"R [$\AA$]")

    axDiffs[i].set_title("Fit at time {} fs".format(dfpc_times[i]))

  
  ####################################
  #####  Plot Molecule Geometry  #####
  ####################################

  axx = fig.add_subplot(grid2[0])
  axy = fig.add_subplot(grid2[1])
  axz = fig.add_subplot(grid2[2])

  axx.axis('off')
  axy.axis('off')
  axz.axis('off')

  axx.set_xlim([-2,2]), axx.set_ylim([-2,2])
  axy.set_xlim([-2,2]), axy.set_ylim([-2,2])
  axz.set_xlim([-2,2]), axz.set_ylim([-2,2])
  gt = orient_CHO(geo_truth[0,:,:])
  for im in range(3):
    axx.add_patch(plt.Circle((gt[im,1], gt[im,2]),
          0.2, color=colors[im]))
    axy.add_patch(plt.Circle((gt[im,0], gt[im,2]),
          0.2, color=colors[im]))
    axz.add_patch(plt.Circle((gt[im,0], gt[im,1]),
          0.2, color=colors[im]))

  for i,itm in enumerate(dfpc_inds):
    j=i+1
    axx = fig.add_subplot(grid2[j*3])
    axy = fig.add_subplot(grid2[j*3+1])
    axz = fig.add_subplot(grid2[j*3+2])

    axx.axis('off')
    axy.axis('off')
    axz.axis('off')

    axx.set_xlim([-2,2]), axx.set_ylim([-2,2])
    axy.set_xlim([-2,2]), axy.set_ylim([-2,2])
    axz.set_xlim([-2,2]), axz.set_ylim([-2,2])
 
    gt = orient_CHO(geo_truth[itm,:,:])
    for im in range(3):
      axx.add_patch(plt.Circle((gt[im,1], gt[im,2]),
            0.2, color=colors[im], alpha=0.5))
      axy.add_patch(plt.Circle((gt[im,0], gt[im,2]),
            0.2, color=colors[im], alpha=0.5))
      axz.add_patch(plt.Circle((gt[im,0], gt[im,1]),
            0.2, color=colors[im], alpha=0.5))

    gh = orient_CHO(geo_history[istp,itm,:,:])
    for im in range(3):
      axx.add_patch(plt.Circle((gh[im,1], gh[im,2]),
            0.2, color=colors[im], fill=False))
      axy.add_patch(plt.Circle((gh[im,0], gh[im,2]),
            0.2, color=colors[im], fill=False))
      axz.add_patch(plt.Circle((gh[im,0], gh[im,1]),
            0.2, color=colors[im], fill=False))



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

  flags_string  = str(FLAGS.flags_into_string())
  flags_hash    = blake2b(flags_string.encode('utf-8'), digest_size=10).hexdigest()
  hist_dir      = os.path.join(parameters["history_dir"], FLAGS.molecule)
  if FLAGS.experiment is not None:
    hist_dir = os.path.join(hist_dir, FLAGS.experiment)
  print("HASH",flags_hash)

  q = get_Q(parameters)
  r = get_R(parameters)
  scat_amps, scat_weights = get_scattering_amps(parameters, q)

  search = "flags-{}".format(flags_hash)
  prev_files = glob.glob(
      os.path.join(hist_dir, search+"*.npy"))
  if len(prev_files) == 0:
    logging.fatal("Cannot find history files with hash {}".format(
        flags_hash))

  prev_file = prev_files[0]
  sInd = prev_file.find("step-") + 5
  fInd = prev_file.find("_", sInd)
  step = int(prev_file[sInd:fInd])
  prefix = "flags-{}_step-{}_".format(flags_hash, step)

  dataFolder = os.path.join(
      parameters["data_dir"],
      FLAGS.molecule)

  with open(os.path.join(hist_dir, prefix+"step_history.npy"), "rb") as file:
    step_history = np.load(file)

  with open(os.path.join(hist_dir, prefix+"loss_history.npy"), "rb") as file:
    loss_history = np.load(file)

  with open(os.path.join(hist_dir, prefix+"dists_history.npy"), "rb") as file:
    dists_history = np.load(file)

  with open(os.path.join(data_dir, "pairwise_distances.npy"), "rb") as file:
    dists_truth = np.load(file)

  fileName = os.path.join(dataFolder, "{}_dataset.npy".format(FLAGS.molecule))
  with open(fileName, "rb") as file:
    geo_truth = np.load(file)

  with open(os.path.join(hist_dir, prefix+"geometry_history.npy"), "rb") as file:
    geo_history = np.load(file)

  Ntime_steps = geo_history.shape[1]
  geo_truth = geo_truth[:Ntime_steps]



  ######################
  #####  Plotting  #####
  ######################

  losses  = [
      "Diffraction Loss", "Pair Correlation Loss",
      "Momentum Loss", "Initial Conditions Loss"]
  colors  = ['r', 'b', 'g']
  atoms   = ['C', 'H', 'O']
  sim_times = np.arange(Ntime_steps)*dt

  dfpc_times  = np.array([3., 6., 10., 45.]) #fs
  dfpc_inds   = []
  for tm in dfpc_times:
    dfpc_inds.append(np.argmin(np.abs(sim_times - tm)))
  dfpc_inds = np.array(dfpc_inds)

  # Check if directory exists
  if FLAGS.experiment is not None:
    plot_dir = os.path.join("../", FLAGS.molecule, FLAGS.experiment, flags_hash)
  else:
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
  pool = Pool(processes=2)
  plt_inds = np.arange(len(step_history)/5)*5
  res = pool.map(partial(make_plot, 
    q=q, r=r,
    scat_weights=scat_weights,
    losses=losses,
    sim_times=sim_times,
    dfpc_times=dfpc_times,
    dfpc_inds=dfpc_inds,
    step_history=step_history,
    loss_history=loss_history,
    geo_history=geo_history,
    geo_truth=geo_truth,
    dists_history=dists_history,
    dists_truth=dists_truth,
    diffraction_truth=diffraction_truth,
    pair_corr_truth=pair_corr_truth,
    diffraction_hist=diffraction_hist,
    pair_corr_hist=pair_corr_hist,
    Ntime_steps=Ntime_steps,
    colors=colors,
    atoms=atoms,
    plot_dir=plot_dir),
    range(len(step_history)))

  print("Finished with flags {}".format(flags_hash))


if __name__ == "__main__":

  app.run(main)

