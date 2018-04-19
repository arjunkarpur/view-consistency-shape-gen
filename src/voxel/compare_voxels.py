import os
import sys
import json
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pred_dir = "../../output/CHAIR/preds/chair-ae3d-long/binary"
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
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.voxels(voxels, edgecolors='k')
        plt.title('Prediction')

        # Get GT
        fp = os.path.join(gt_dir, "%s.mat" % id_)
        mat = sio.loadmat(fp)
        gt_voxels = mat['data']
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.voxels(gt_voxels, facecolors='red', edgecolors='k')
        plt.title('Ground Truth')

        def on_move(event):
            if event.inaxes == ax1:
                ax2.view_init(elev=ax1.elev, azim=ax1.azim)
            elif event.inaxes == ax2:
                ax1.view_init(elev=ax2.elev, azim=ax2.azim)
            else:
                return
            fig.canvas.draw_idle()

        # Print to screen
        print str(mat['id'][0])
        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
        plt.show()
    break
