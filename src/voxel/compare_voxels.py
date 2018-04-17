import os
import sys
import json
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pred_dir = "../../output/CHAIR/preds/test-ae3d/binary"
gt_dir = "../../data/CHAIR/mat/"

for root, dirs, files in os.walk(pred_dir):
    random.shuffle(files)
    for f in files:
        fp = os.path.join(dir_, f)
        mat = sio.loadmat(fp)
        id_ = mat['id']
        fig = plt.figure()

        # Get pred
        fig.subplot(1,2,1)
        voxels = mat['data']
        ax = fig.gca(projection='3d')
        ax.voxels(voxels, edgecolors='k')

        # Get GT
        fig.subplot(1,2,2)
        fp = os.path.join(gt_dir, "%s.mat" % id_)
        mat = sio.loadmat(fp)
        gt_voxels = mat['data']
        ax = fig.gca(projection='3d')
        ax.voxels(gt_voxels, edgecolors='k')

        # Print to screen
        print mat['id']
        plt.show()
    break
