import os
import sys
import json
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pred_dir = "../../output/CHAIR/preds/chair-ae3d/binary"
gt_dir = "../../data/CHAIR/mat/"

for root, dirs, files in os.walk(pred_dir):
    random.shuffle(files)
    for f in files:
        fp = os.path.join(pred_dir, f)
        mat = sio.loadmat(fp)
        id_ = str(mat['id'][0])
        fig = plt.figure(figsize=plt.figaspect(.5))
        fig.suptitle(str(mat['id'][0]))

        # Get pred
        voxels = mat['data']
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.voxels(voxels, edgecolors='k')
        plt.title('Prediction')

        # Get GT
        fp = os.path.join(gt_dir, "%s.mat" % id_)
        mat = sio.loadmat(fp)
        gt_voxels = mat['data']
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.voxels(gt_voxels, facecolors='red', edgecolors='k')
        plt.title('Ground Truth')

        # Print to screen
        print str(mat['id'][0])
        plt.show()
    break
