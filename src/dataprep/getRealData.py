import os
import sys
import png
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
DEBUG = False

# Filepath parameters
version = '_v1'
MESHES_LIST_PATH = "./meshes-chair125.txt"
DATA_PATH = os.path.join("/", "hdd", "zxy", "Datasets", "Redwood_3DScan_Chairs", "CHAIR")
RGB_PATH = os.path.join("/", "home", "arjun", "research", "thesis", "shape-gen", "data", "RedwoodRGB_Chair")
PROCESSED_PATH = os.path.join(RGB_PATH, "processed")
SAVE_PATH = os.path.join(PROCESSED_PATH, 'images{}/'.format(version))
IMS_PATH = os.path.join(DATA_PATH, "ims")
RGBIM_PATH = os.path.join(RGB_PATH, "ims")
MESHES_PATH = os.path.join(DATA_PATH, "meshes")
KEYPOINT_PATH = os.path.join(MESHES_PATH, "keypoints") 

if not os.path.exists(SAVE_PATH):
  os.mkdir(SAVE_PATH)

# Read list of meshes to perform annotation for
mesh_names = []
in_f = open(MESHES_LIST_PATH, 'r')
lines = in_f.readlines()
in_f.close()
for l in lines:
    l = l.strip()
    mesh_names.append(l.split(".")[0])

# Parameters for input/output
num_models = len(mesh_names)
J = 10
H = 480
W = 640
S = 525
OUTSIZE = 224
RATIO = 0.0
oo = 1e6
D = 20
DEPTH_SHIFT = 1000
P = 10
RT = 0.05

# Begin image processing and annotation
info = open(os.path.join(PROCESSED_PATH, 'info{}.txt'.format(version)), 'w')
model_count = 0
for model_id in xrange(1, num_models+1):
    # Gather model name and corresponding 3D keypoints
    model_name = mesh_names[model_id-1]
    print "Processing model %i/%i:\t%s" % (model_id, num_models, model_name)
    keypoints_path = os.path.join(KEYPOINT_PATH, "%s_picked_points.pp" % model_name)
    points = np.zeros((J,4))
    try:
        f = open(keypoints_path)
    except:
        print "%s - KEYPOINT FILE NOT FOUND" % model_name
        continue
    for j, line in enumerate(f):
        if j >= 8 and j <= 17:
            try:
                tmp = line[line.find('x="') + 3:]
                points[j - 8, 0] = float(tmp[:tmp.find('"')])
                tmp = line[line.find('y="') + 3:]
                points[j - 8, 1] = float(tmp[:tmp.find('"')])
                tmp = line[line.find('z="') + 3:]
                points[j - 8, 2] = float(tmp[:tmp.find('"')])
                points[j - 8, 3] = 1
                fail = False
            except:
                print "%s - ERROR READING KEYPOINT FILE" % model_name
                fail = True
                break
    if fail:
        continue
  
    scale = ((points[:, 0].max() - points[:, 0].min()) ** 2 + \
            (points[:, 1].max() - points[:, 1].min()) ** 2 + \
            (points[:, 2].max() - points[:, 2].min()) ** 2) ** 0.5
    c = points[2:, 2].mean() * DEPTH_SHIFT
    SCALE = int(scale * DEPTH_SHIFT)

    # Gather list of frames names for current model
    depth_ims_dir = os.path.join(IMS_PATH, model_name, "depth_uint8")
    RGB_ims_dir = os.path.join(RGBIM_PATH, model_name, "rgb")

    frames__ = []
    dic = {}
    for f in os.listdir(depth_ims_dir):
        if (os.path.isfile(os.path.join(depth_ims_dir, f)) and f.split(".")[-1] == "png"):
            frames__.append(f.split("-")[0])
            dic[f.split("-")[0]] = f.split(".")[0]
    frames = [f.split(".")[0] for f in os.listdir(RGB_ims_dir) if
        (os.path.isfile(os.path.join(RGB_ims_dir, f)) and f.split(".")[-1] == "jpg") and (f.split("-")[0] in frames__)]
    num_frames = len(frames)

    # Crop and annotate each frame
    frame_count = 1
    for frame_name in frames:
    
        # Get frame details (name, depth im, pose)
        #depth_path = os.path.join(IMS_PATH, model_name, "depth_uint8", "%s.png" % frame_name)
        RGB_path = os.path.join(RGBIM_PATH, model_name, "rgb", "%s.jpg" % frame_name)
        traj_path = os.path.join(IMS_PATH, model_name, "xf", "%s.xf" % dic[frame_name.split('-')[0]])
        try:
            pose = np.loadtxt(traj_path)
        except:
            print "%s - POSE FILE DOESNT EXIST" % frame_name
            continue


        # Calc new points
        old_points = np.dot(np.linalg.inv(pose), points.transpose(1,0)).transpose(1,0)[:, :3]
        new_points = np.zeros((J, 4))
        vis = np.zeros((J), np.int32)
        for j in xrange(J):
            new_points[j,0] = old_points[j,0] / old_points[j,2] * S + 320
            new_points[j,1] = old_points[j,1] / old_points[j,2] * S + 240
            new_points[j,2] = old_points[j,2] * DEPTH_SHIFT
            u, v, d = int(new_points[j, 0]), int(new_points[j, 1]), int(new_points[j, 2])
    
        RGB = cv2.imread(RGB_path)
        img = RGB.copy()
        t,d = int(new_points[:,0].min()), int(new_points[:,0].max())
        l,r = int(new_points[:,1].min()), int(new_points[:,1].max())

        if l > 0 - RATIO * W and r < W + RATIO * W and t > 0 - RATIO * H and d < H + RATIO * H:
            # Valid frame, crop/copy frame
            wc = (l + r) / 2
            hc = (t + d) / 2
            pad = D#max(0, SCALE / 4 - max(r - l, d - t) / 2)
            s = max(r - l, d - t) / 2 + pad
            ss = min(r - l, d - t) / 2 + pad
            new_img = np.ones((H+2*s, W+2*s, 3), dtype = np.uint8) * 255
            wc += s
            hc += s
            pad2 = D
            #print 'pad2', pad2
            try:
                new_img[s:H + s, s:W + s] = img.copy()
                out = np.ones((2*s + 2 * pad2, 2*s + 2 * pad2, 3), dtype = np.uint8) * 255
                if r-l > d-t:
                    out[pad2:-pad2, pad2 + s-ss: pad2 + s+ss] = new_img[wc - s: wc+s, hc-ss: hc+ss].copy()
                else:
                    out[pad2 + s-ss: pad2 + s+ss, pad2:-pad2] = new_img[wc - ss: wc + ss, hc - s: hc + s].copy()
                out = cv2.resize(out, (OUTSIZE, OUTSIZE))
            except:
                print "%s - ERROR WRITING OUTPUT IMAGE" % frame_name
                continue

            # Save image to .bmp
            out_fp = os.path.join(SAVE_PATH, "%s_%s.bmp" % (model_count, frame_count))
            print "Writing %s_%s.bmp" % (model_count, frame_count)
            cv2.imwrite(out_fp, out)
            frame_count += 1
    if frame_count > 20:
        model_count += 1
    else:
        continue
        info.write('{} {}\n'.format(model_count - 1, frame_count))  
