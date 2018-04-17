import os
import sys
import json
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dir_ = "../../output/CHAIR/preds/test-ae3d/binary"

for root, dirs, files in os.walk(dir_):
    random.shuffle(files)
    for f in files:
        fp = os.path.join(dir_, f)
        mat = sio.loadmat(fp)
        voxels = mat['data']
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, edgecolors='k')
        print mat['id']
        plt.show()
    break
