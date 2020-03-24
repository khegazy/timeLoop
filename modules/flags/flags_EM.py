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
      "Nsteps", 50000,
      "Number of fitting steps to take.")

  flags.DEFINE_string(
      "optimizer", "SGD",
      "Optimizer for fitting.")

  flags.DEFINE_float(
      "SGD_LR", 1e-3,
      "Learning rate for SGD.")

  flags.DEFINE_string(
      "loss_type", "EM",
      "Indicate which loss to use for diffraction and PC: L2=0, EM=1.")

  flags.DEFINE_float(
      "scale_diffraction_loss", 5,#1e-4,
      "Scaling hyperparameter for the diffraction loss contribution.")

  flags.DEFINE_float(
      "scale_pairCorr_loss", 4e-2,
      "Scaling hyperparameter for the pair correlation loss contribution.")

  flags.DEFINE_float(
      "scale_init_zero_loss", 1,
      "Scaling hyperparameter for the init zero loss contribution.")

  flags.DEFINE_float(
      "scale_velocity_loss", 1e-7,
      "Scaling hyperparameter for the velocity loss contribution.")

  flags.DEFINE_float(
      "velocity_scale", 0.1,
      "Characteristic velocity in A/fs for the velocity loss.")

  flags.DEFINE_boolean(
      "debugging", False,
      "Debugging option that defines what debugging study to do.")

  flags.DEFINE_string(
      "debugging_opt", "init_to_noisy_data,calc_grad_mags",
      "Debugging option that defines what debugging study to do.")

  return FLAGS

