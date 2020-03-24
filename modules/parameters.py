import os


xSection_folder = "./data/scatteringXsections"

parameters = {
    "atoms" : {
      "hydrogen"  : {
              'ind'       : 0,
              'xSection'  : os.path.join(xSection_folder, "hydrogen_dcs.dat"),
              'N'         : 0
            },
      "carbon"    : {
              'ind'       : 1,
              'xSection'  : os.path.join(xSection_folder, "carbon_dcs.dat"),
              'N'         : 0
            },
      "oxygen"    : {
              'ind'       : 2,
              'xSection'  : os.path.join(xSection_folder, "oxygen_dcs.dat"),
              'N'         : 0
            },
      "nitrogen"  : {
              'ind'       : 3,
              'xSection'  : os.path.join(xSection_folder, "nitrogen_dcs.dat"),
              'N'         : 0
            },
      "iodine"    : {
              'ind'       : 4,
              'xSection'  : os.path.join(xSection_folder, "iodine_dcs.dat"),
              'N'         : 0
            }
      },
    "atom_types"  : ["carbon", "hydrogen", "oxygen"],
    "dt"          : 0.25,
    "q"           : (10, 250),
    "r"           : (5, 100),
    "data_dir"    : "data",
    "log_every"   : 10,
    "save_every"  : 250,#250000,
    "history_dir" : os.path.join("output", "history")
}


def get_parameters():
  if "atom_types" in parameters:
    for atm in parameters["atom_types"]:
      parameters["atoms"][atm]['N'] += 1

  return parameters
