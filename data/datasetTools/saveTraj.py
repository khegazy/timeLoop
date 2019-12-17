import numpy as np
import mdtraj
import os


molecule = "CHO"

baseFolder  = "../" + molecule 
fileName    = os.path.join(baseFolder, "scripts", "{}.lammpstrj".format(molecule))
res = mdtraj.formats.LAMMPSTrajectoryFile(fileName)
xyz = res.read_as_traj(None).xyz
print(xyz[0,:,:])
xyz *= 10
xyz *= 50

#dists = np.concatenate([
#    np.expand_dims(xyz[:,:,0] - xyz[:,:,1], axis=-1),
#    np.expand_dims(xyz[:,:,0] - xyz[:,:,2], axis=-1),
#    np.expand_dims(xyz[:,:,1] - xyz[:,:,2], axis=-1)], axis=-1)

outFileName = os.path.join(baseFolder, "{}_dataset.npy".format(molecule))
with open(outFileName, "wb") as file:
  np.save(file, xyz)


