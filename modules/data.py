import jax as jax
import jax.numpy as np

import sys
import os
import numpy as onp
import matplotlib.pyplot as plt



def get_data(FLAGS, parameters):

  dataFolder = os.path.join(
      parameters["data_dir"],
      FLAGS.molecule)

  with open(os.path.join(dataFolder, "{}_dataset.npy".format(FLAGS.molecule)), "rb") as file:
    atom_positions    = np.load(file)[:200]
    initial_geometry  = atom_positions[0,:,:]

  with open(os.path.join(dataFolder, "pairwise_distances.npy"), "rb") as file:
    pairDists_data = np.load(file)[:200]

  with open(os.path.join(dataFolder, "angles.npy"), "rb") as file:
    angles_data = None #np.load(file)[:200]

  with open(os.path.join(dataFolder, "mmd_diffraction.npy"), "rb") as file:
    mmd_data = np.load(file)[:200]

  with open(os.path.join(dataFolder, "pair_correlations.npy"), "rb") as file:
    pairCorr_data = np.load(file)[:200]

  return initial_geometry, atom_positions,\
      pairDists_data, angles_data,\
      mmd_data, pairCorr_data
