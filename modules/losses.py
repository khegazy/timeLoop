import jax as jax
import jax.numpy as np

from jax.api import jit

def earth_movers_distance(p, q):
  #assert len(p) == len(q)

  p = p / np.sum(p)
  q = q / np.sum(q)

  dpq = p - q

  return np.mean(np.sum(np.abs(np.cumsum(dpq, axis=-1)), axis=-1))


def wasserstein_distance(sim, data, x):
  sim = sim[0,:]
  data = data[0,:]

  data_cdf = np.cumsum(data, axis=-1)
  data /= data_cdf[-1]
  data_cdf /= data_cdf[-1]

  sim_cdf = np.cumsum(sim, axis=-1)
  sim /= sim_cdf[-1]
  sim_cdf /= sim_cdf[-1]

  print("SHAPES", data.shape, data_cdf.shape)
  print("SHAPES", x.shape)
  finvGx = []
  for d in data_cdf:
    ind = np.argmin(np.abs(d - sim_cdf))
    finvGx.append(x[ind])
  finvGx = np.array(finvGx)

  print("SHAPES", finvGx.shape)

  return np.sum(np.abs(finvGx - x )*data)



def L2_loss(fit, truth):
  diff = fit - truth
  print(np.any(np.isnan(diff)))
  return np.mean(np.sum(np.square(fit - truth), axis=-1))
