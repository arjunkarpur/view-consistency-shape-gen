import os
import sys
import json
import random
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D

pred_dir = "../../output/CHAIR/preds/joint-train-only-1/binary/0_4"
im_dir = "../../data/CHAIR/renderings/"
gt_dir = "../../data/CHAIR/mat/"

pred_files = []
for root, dirs, files in os.walk(pred_dir):
    for d in dirs:
        for _,_,fs in os.walk(os.path.join(pred_dir, d)):
            pred_files += [os.path.join(pred_dir,d,f) for f in fs]
            break
    break
random.shuffle(pred_files)
print len(pred_files)

for f in pred_files:
    mat = sio.loadmat(f)
    id_ = str(mat['id'][0])
    fig = plt.figure(figsize=plt.figaspect(.33))
    fig.suptitle(str(mat['id'][0]).split("/")[-1])

    # Get input image
    im_fp = im_dir + "/".join(f.split("/")[-2:]).split(".")[0] + ".jpg"
    im = mpimg.imread(im_fp)
    print im.shape
    y,x,_ = im.shape
    startx = x//2-(224//2)
    starty = y//2-(224//2)
    im = im[starty:starty+224,startx:startx+224,:]
    ax0 = fig.add_subplot(1, 3, 1)
    plt.imshow(im)

    # Get pred
    voxels = mat['data']
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    ax1.voxels(voxels, edgecolors='k')
    plt.title('Prediction')

    # Get GT
    obj_id = id_.split("/")[-1].split("_")[1]
    fp = os.path.join(gt_dir, "%s.mat" % obj_id)
    mat = sio.loadmat(fp)
    gt_voxels = mat['data']
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
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
