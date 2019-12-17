import os


xSection_folder = "/media/kareem/09F434A51ACAE825/Research/experiments/timeLoop/data/scatteringXsections"

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
    "atom_types": ["carbon", "hydrogen", "oxygen"],
    "dt"        : 0.25,
    "q"         : (10, 500),
    "r"         : (5, 250)
}


def get_parameters():
  if "atom_types" in parameters:
    for atm in parameters["atom_types"]:
      parameters["atoms"][atm]['N'] += 1

  return parameters
