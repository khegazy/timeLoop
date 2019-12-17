import jax as jax
import jax.numpy as np

from jax.api import jit
from jax.api import vmap

import os
import sys
import time
from enum import Enum

import numpy as onp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from modules.parameters import get_parameters
from modules.diffraction import *


if __name__ == "__main__":

  molecule = "CHO"
  baseFolder = os.path.join("../", molecule)
  rng = jax.random.PRNGKey(5)

  params = get_parameters()

  #############################
  #####  Import xyz data  #####
  #############################

  fileName = os.path.join(baseFolder, "{}_dataset.npy".format(molecule))
  with open(fileName, "rb") as file:
    xyz_data = np.load(file)


  #################################
  #####  Calculate Distances  #####
  #################################

  dists   = dist_metric_td(xyz_data)
  angles  = angle_metric_td(xyz_data)
  print(angles)


  ###################################
  #####  Calculate Diffraction  #####
  ###################################

  q = get_Q(params)
  scat_amps, scat_weights = get_scattering_amps(params, q)

  mm_diffraction, mol_diffraction, atomic_diffraction = diffraction(
      dists, q, scat_weights)


  ########################################
  #####  Calculate Pair Correlation  #####
  ########################################

  r = get_R(params)
  pair_corr = calc_pair_corr(q, r, mm_diffraction, 3.5)


  ############################
  #####  Saving Results  #####
  ############################
  
  with open(os.path.join(baseFolder, "initial_geometry.npy"), "wb") as file:
    np.save(file, xyz_data[0,:,:])

  with open(os.path.join(baseFolder, "pairwise_distances.npy"), "wb") as file:
    np.save(file, dists)

  with open(os.path.join(baseFolder, "angles.npy"), "wb") as file:
    np.save(file, angles)

  with open(os.path.join(baseFolder, "mmd_diffraction.npy"), "wb") as file:
    np.save(file, mm_diffraction)

  with open(os.path.join(baseFolder, "atom_diffraction.npy"), "wb") as file:
    np.save(file, atomic_diffraction)

  with open(os.path.join(baseFolder, "pair_correlations.npy"), "wb") as file:
    np.save(file, pair_corr)

  for i in range(100):
    ind = i #jax.random.randint(rng, (1,), 0, data_dists.shape[0])[0]
    print("plotting",i,ind)
    fig, ax = plt.subplots(2, 1)

    ax[0].plot(q, mm_diffraction[ind,:])
    ax[0].set_xlim([q[0], q[-1]])
    ax[0].set_xlabel(r"Q [$\AA^{-1}$]")
    ax[1].plot(r, pair_corr[ind,:])
    ax[1].set_xlim([r[0], r[-1]])
    ax[1].set_xlabel(r"R [$\AA$]")

    fig.tight_layout()
    fig.savefig("./plots/data_{}.png".format(ind))
    plt.close()
