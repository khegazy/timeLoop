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
      "leFlag", "I am le Flag",
      "test")

  flags.DEFINE_string(
      "molecule", "CHO",
      "Name of the molecule under investigation.")

  flags.DEFINE_string(
      "data_dir", "data",
      "Folder where the dataset resides.")

  flags.DEFINE_integer(
      "log_every", 50,
      "Log fitting progress every N steps.")

  flags.DEFINE_string(
      "history_dir", "output/history",
      "Folder where the fitting history resides.")

  flags.DEFINE_float(
      "scale_diffraction_loss", 10,
      "Scaling hyperparameter for the diffraction loss contribution.")

  flags.DEFINE_float(
      "scale_pairCorr_loss", 0.01,
      "Scaling hyperparameter for the pair correlation loss contribution.")

  flags.DEFINE_float(
      "scale_init_zero_loss", 1,
      "Scaling hyperparameter for the init zero loss contribution.")

  flags.DEFINE_float(
      "scale_velocity_loss", 0.000001,
      "Scaling hyperparameter for the velocity loss contribution.")

  flags.DEFINE_float(
      "velocity_scale", 0.1,
      "Characteristic velocity in A/fs for the velocity loss.")

  return FLAGS

