import jax as jax
import jax.numpy as np
import numpy as onp

from jax.api import jit
from jax.api import vmap

from jax_md import space
from jax_md import minimize
from jax_md import simulate
from jax_md import space
from jax_md import energy
from jax_md import quantity

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#############################################
#####  Calculate Diffraction Variables  #####
#############################################

def get_Q(params):
  qMax, NqBins = params["q"]
  q = np.linspace(0, qMax, NqBins)
  return q + (q[1] - q[0])/2.

def get_R(params):
  rMax, NrBins = params["r"]
  r = np.linspace(0, rMax, NrBins)
  return r + (r[1] - r[0])/2.


###################################
#####  Distance Calculations  #####
###################################

displacement, shift = space.free()
metric = space.metric(displacement)
metric = space.map_product(metric)

def dist_metric(R):
  return metric(R,R)

dist_metric_td = vmap(dist_metric, (0))


#####  Angles  #####

def calc_angle(dR_12, dR_23):
  return np.dot(dR_12, dR_23)\
      /(np.sqrt(np.dot(dR_12, dR_12))*np.sqrt(np.dot(dR_23, dR_23)))

calc_angle_vm = vmap(vmap(vmap(calc_angle, (0, None)), (None, 0)))
calc_angle_vm_td = vmap(calc_angle_vm, (0, 0))

def angle_metric(R):
  dR = metric(R,R)
  return calc_angle_vm(dR, dR)

def angle_metric_td(R):
  dR = dist_metric_td(R)
  calc_angle_vm_td(dR, dR)


#############################
#####  Data Simulation  #####
#############################

def get_scattering_amps(params, q):

  xSections = []

  for atm in params["atoms"].keys():

    angStr = []
    sctStr = []

    with open(params["atoms"][atm]["xSection"], 'r') as file:
      ind=0
      for line in file:
        if ind < 31:
          ind += 1
          continue

        angStr.append(line[2:11])
        sctStr.append(line[39:50])

    angs = onp.array(angStr).astype(onp.float64)
    scts = onp.sqrt(onp.array(sctStr).astype(onp.float64))

    xSections.append(interp1d(angs, scts, 'cubic'))

  el_energy = 3.7*10**6
  C_AU      = 1./0.0072973525664
  ev_to_au  = 0.0367493

  deBrog = 2*np.pi*C_AU/np.sqrt((el_energy*ev_to_au + C_AU**2)**2 - C_AU**4)
  deBrog /= 1e-10/5.291772108e-11 # au to angs
  eval_angles = 2*np.arcsin(q*deBrog/(4*np.pi))

  scat_amps = []
  for i in np.arange(len(params["atoms"].keys())):
    scat_amps.append(np.expand_dims(xSections[i](eval_angles), axis=0))
  scat_amps = np.concatenate(scat_amps, axis=0)

  if "atom_types" in params:
    atom_inds = []
    for atm in params["atom_types"]:
      atom_inds.append(params["atoms"][atm]["ind"])

    scat_terms = scat_amps[atom_inds]
    scat_terms = np.expand_dims(scat_terms, 0)*np.expand_dims(scat_terms, 1)

  return scat_amps, scat_terms


def diffraction(dists, q, scat_weights):

  diffr_inds = np.logical_not(
      np.eye(scat_weights.shape[0], dtype=int).astype(bool))

  arg = np.expand_dims(dists[:,diffr_inds], axis=-1)*q
  calc = scat_weights[diffr_inds]*np.sin(arg)/arg

  atomic = np.sum(scat_weights[np.logical_not(diffr_inds)], axis=0)
  mol     = np.sum(calc, axis=1)/2.
  mmd     = mol*q/atomic

  return mmd, mol, atomic


def get_sine_transform_matrix(q, r):

  return np.sin(np.expand_dims(r, axis=0)*np.expand_dims(q, axis=-1))


def smooth_diffraction(diffrxn, q, smooth_std):
  filter = np.expand_dims(
      np.exp(-0.5*(q/smooth_std)**2),
      axis=0)
  return diffrxn*filter, filter


@jit
def calc_pair_corr(q, r, diffrxn, smooth_std):
  sine_trans = get_sine_transform_matrix(q, r)
  smooth, _ = smooth_diffraction(diffrxn, q, smooth_std)
  return np.matmul(smooth, sine_trans)

