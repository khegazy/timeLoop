import os
import sys
import numpy as np
import matplotlib.pyplot as plt


##################################
#####  Get History and Data  #####
##################################

molecule = "CHO"
dt = 0.25
hist_dir = "../../output/history/" + molecule
data_dir = "../../data/" + molecule

with open(os.path.join(hist_dir, "step_history.npy"), "rb") as file:
  step_history = np.load(file)

with open(os.path.join(hist_dir, "loss_history.npy"), "rb") as file:
  loss_history = np.load(file)

with open(os.path.join(hist_dir, "dists_history.npy"), "rb") as file:
  dists_history = np.load(file)

with open(os.path.join(data_dir, "pairwise_distances.npy"), "rb") as file:
  dists_truth = np.load(file)



#########################
#####  Plot Losses  #####
#########################

losses = [
    "Diffraction Loss", "Pair Correlation Loss",
    "Velocity Loss", "Initial Conditions Loss"]
print(loss_history.shape)
fig, ax = plt.subplots()
for i,lss in enumerate(losses):
  ax.plot(step_history, loss_history[:,i], label=lss)
ax.plot(step_history, np.sum(loss_history, axis=1), '-k', label="Total Loss")
ax.set_xlim(step_history[0], step_history[-1])
ax.legend()
fig.savefig(os.path.join("..", molecule, "loss_history.png"))
plt.close()



#####################################
#####  Plot Pairwise Distances  #####
#####################################

print(dists_history.shape, dists_truth.shape)
colors = ['r', 'b', 'g']
atoms = ['C', 'H', 'O']
tCut = 400
tm = np.arange(tCut)*dt
print("size",step_history.shape)
for stp in range(step_history.shape[0]):
  ind = 0
  fig, ax = plt.subplots()
  for i in range(3):
    for j in range(i+1, 3):
      ax.plot(
          tm,
          dists_truth[:400,i,j],
          c=colors[ind],
          label="{} - {}".format(atoms[i], atoms[j]))
      ax.plot(
          tm,
          dists_history[stp,:400,i,j],
          c=colors[ind], linestyle=":")
      ind += 1
  ax.set_xlim([tm[0], tm[-1]])
  ax.set_xlabel("Time [fs]")
  ax.set_ylabel(r"Pair Distance [$\AA$]")
  ax.legend()
  fig.tight_layout()
  fig.savefig(os.path.join("..", molecule, "dists_train-{}.png".format(stp)))
  plt.close()
