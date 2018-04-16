import sys
import json
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

f = open("../../data/CHAIR/json/name_to_id.json", 'r')
names = (json.loads((f.readlines()[0]).strip())).keys()
f.close()
random.shuffle(names)

for name in names:
    voxels = sio.loadmat(str("../../data/CHAIR/mat/%s.mat" % name))['data']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, edgecolors='k')
    plt.show()

