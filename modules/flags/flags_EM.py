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
<<<<<<< HEAD
      "experiment", "time_dep_weights",
      "Name of the experiment.")

  flags.DEFINE_string(
=======
>>>>>>> 3c797c16f759928cf2cb6f1adeaeee7f33375f50
      "molecule", "CHO",
      "Name of the molecule under investigation.")

  flags.DEFINE_integer(
<<<<<<< HEAD
      "Nsteps", 802,
=======
      "Nsteps", 50000,
>>>>>>> 3c797c16f759928cf2cb6f1adeaeee7f33375f50
      "Number of fitting steps to take.")

  flags.DEFINE_string(
      "optimizer", "SGD",
      "Optimizer for fitting.")

  flags.DEFINE_float(
<<<<<<< HEAD
      "SGD_LR", 1e-6,
=======
      "SGD_LR", 1e-3,
>>>>>>> 3c797c16f759928cf2cb6f1adeaeee7f33375f50
      "Learning rate for SGD.")

  flags.DEFINE_string(
      "loss_type", "EM",
      "Indicate which loss to use for diffraction and PC.")

  flags.DEFINE_float(
<<<<<<< HEAD
      "scale_diffraction_loss", 1e-4,
      "Scaling hyperparameter for the diffraction loss contribution.")

  flags.DEFINE_float(
      "scale_pairCorr_loss", 1e1,
      "Scaling hyperparameter for the pair correlation loss contribution.")

  flags.DEFINE_float(
      "scale_init_zero_loss", 1e3,
=======
      "scale_diffraction_loss", 1e1,#1e-4,
      "Scaling hyperparameter for the diffraction loss contribution.")

  flags.DEFINE_float(
      "scale_pairCorr_loss", 6e3,
      "Scaling hyperparameter for the pair correlation loss contribution.")

  flags.DEFINE_float(
      "scale_init_zero_loss", 1e2,
>>>>>>> 3c797c16f759928cf2cb6f1adeaeee7f33375f50
      "Scaling hyperparameter for the init zero loss contribution.")

  flags.DEFINE_float(
      "scale_momentum_loss", 3e-6,
      "Scaling hyperparameter for the momentum loss contribution.")

  flags.DEFINE_float(
      "momentum_scale", 0.05,
      "Characteristic momentum in A*AU/fs for the momentum loss.")

<<<<<<< HEAD
  flags.DEFINE_float(
      "scale_force_loss", 3e-4,
      "Scaling hyperparameter for the force loss contribution.")

  flags.DEFINE_float(
      "force_scale", 0.25,
      "Characteristic momentum in A*AU/fs**2 for the momentum loss.")

  flags.DEFINE_float(
      "scale_expect_pos_loss", 1e4,
      "Characteristic momentum in A*AU/fs**2 for the momentum loss.")

=======
>>>>>>> 3c797c16f759928cf2cb6f1adeaeee7f33375f50
  flags.DEFINE_boolean(
      "debugging", False,
      "Debugging option that defines what debugging study to do.")

  flags.DEFINE_string(
<<<<<<< HEAD
      "debugging_opt", None,# "calc_grad_mags",#"init_to_noisy_data,calc_grad_mags",
=======
      "debugging_opt", "calc_grad_mags",#"init_to_noisy_data,calc_grad_mags",
>>>>>>> 3c797c16f759928cf2cb6f1adeaeee7f33375f50
      "Debugging option that defines what debugging study to do.")

  return FLAGS

