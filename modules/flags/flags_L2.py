from absl import flags

import jax as jax
import jax.numpy as np

from jax.api import jit

import sys
import os
import numpy as onp
import matplotlib.pyplot as plt


def get_flags():
  FLAGS = flags.FLAGS

  flags.DEFINE_string(
      "molecule", "CHO",
      "Name of the molecule under investigation.")

  flags.DEFINE_integer(
      "Nsteps", 150000,
      "Number of fitting steps to take.")

  flags.DEFINE_string(
      "optimizer", "SGD",
      "Optimizer for fitting.")

  flags.DEFINE_float(
      "SGD_LR", 5e-2,
      "Learning rate for SGD.")

  flags.DEFINE_string(
      "loss_type", "L2",
      "Indicate which loss to use for diffraction and PC: L2=0, EM=1.")

  flags.DEFINE_float(
      "scale_diffraction_loss", 10,
      "Scaling hyperparameter for the diffraction loss contribution.")

  flags.DEFINE_float(
      "scale_pairCorr_loss", 0.02,
      "Scaling hyperparameter for the pair correlation loss contribution.")

  flags.DEFINE_float(
      "scale_init_zero_loss", 1,
      "Scaling hyperparameter for the init zero loss contribution.")

  flags.DEFINE_float(
      "scale_velocity_loss", 1e-7,#0.001,
      "Scaling hyperparameter for the velocity loss contribution.")

  flags.DEFINE_float(
      "velocity_scale", 0.1,
      "Characteristic velocity in A/fs for the velocity loss.")

  flags.DEFINE_boolean(
      "debugging", False,
      "Generic debugging option.")

  flags.DEFINE_string(
      "debugging_opt", "init_to_noisy_data",
      "Debugging option that defines what debugging study to do.")

  return FLAGS

