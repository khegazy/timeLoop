import jax as jax
import jax.numpy as np

from jax.api import jit

def earth_movers_distance(p, q):
  #assert len(p) == len(q)

  p = p / np.sum(p)
  q = q / np.sum(q)

  dpq = p - q

  return np.sum(np.abs(np.cumsum(dpq)))


def L2_loss(fit, truth):
  diff = fit - truth
  print(np.any(np.isnan(diff)))
  return np.mean(np.square(fit - truth))
