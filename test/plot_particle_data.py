import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

data_file = r'C:\src\mirror\build\test\particle_trajectories.csv'
data = np.loadtxt(data_file, delimiter=",", skiprows=1)

nParticles = int(data.shape[1] / 6)
print('Found {} particles'.format(nParticles))

fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(nParticles):
    ax.plot(data[:,6*i], data[:,6*i+1], data[:,6*i+2])

plt.show()