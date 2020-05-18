import os

import jax as jax
import jax.numpy as np


xSection_folder = "./data/scatteringXsections"

parameters = {
    "atom_types"  : ["carbon", "hydrogen", "oxygen"],
    "dt"          : 0.25,
    "q"           : (10, 250),
    "r"           : (5, 100),
    
    "atoms" : {
      "hydrogen"  : {
              'ind'       : 0,
              'xSection'  : os.path.join(xSection_folder, "hydrogen_dcs.dat"),
              'N'         : 0,
              'mass'      : 1
            },
      "carbon"    : {
              'ind'       : 1,
              'xSection'  : os.path.join(xSection_folder, "carbon_dcs.dat"),
              'N'         : 0,
              'mass'      : 12
            },
      "oxygen"    : {
              'ind'       : 2,
              'xSection'  : os.path.join(xSection_folder, "oxygen_dcs.dat"),
              'N'         : 0,
              'mass'      : 16
            },
      "nitrogen"  : {
              'ind'       : 3,
              'xSection'  : os.path.join(xSection_folder, "nitrogen_dcs.dat"),
              'N'         : 0,
              'mass'      : 14
            },
      "iodine"    : {
              'ind'       : 4,
              'xSection'  : os.path.join(xSection_folder, "iodine_dcs.dat"),
              'N'         : 0,
              'mass'      : 126.9
            }
      },
    "data_dir"    : "data",
    "log_every"   : 1,
    "save_every"  : 100,#250000,
    "history_dir" : os.path.join("output", "history")
}


def get_parameters():
  if "atom_types" in parameters:
    parameters["atom_masses"] = []
    for atm in parameters["atom_types"]:
      parameters["atoms"][atm]['N'] += 1
      parameters["atom_masses"].append(parameters["atoms"][atm]['mass'])
    parameters["atom_masses"] = np.array(parameters["atom_masses"])

  return parameters
