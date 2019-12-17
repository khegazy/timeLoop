import jax as jax
import jax.numpy as np



def earth_movers_distance(p, q):
  #assert len(p) == len(q)

  p = p / np.sum(p)
  q = q / np.sum(q)

  dpq = p - q

  return np.sum(np.abs(np.cumsum(dpq)))
