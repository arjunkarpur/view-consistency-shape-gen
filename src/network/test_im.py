import numpy as np
import os
import sys
import math
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision as tv
import scipy.io as scio

# Imports from src files
from datasets import ImageVoxelDataset
from models import AE_3D

GPU = True
MULTI_GPU = True
MODELS_OBJ_CLASS = "RedwoodRGB_Chair" #use ae,im network models trained on this dataset
OBJ_CLASS = "RedwoodRGB_Chair" #test on this dataset
NAME = "real_data_test"
DATA_BASE_DIR = "../../data/%s" % OBJ_CLASS
IN_AE_WEIGHTS_FP = "../../output/%s/models/%s/joint_ae3d.pt" % (MODELS_OBJ_CLASS, NAME)
IN_IM_WEIGHTS_FP = "../../output/%s/models/%s/joint_im.pt" % (MODELS_OBJ_CLASS, NAME)
OUTPUT_DIR = "../../output/%s/preds/%s" % (OBJ_CLASS, NAME)
OUTPUT_PROB_DIR = "%s/prob" % OUTPUT_DIR
OUTPUT_BINARY_DIR = "%s/binary" % OUTPUT_DIR

VOXEL_RES = 20
EMBED_SIZE = 64
BATCH_SIZE = 192
BIN_THRESHES = \
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
    print "[%s]\t %s" % (datetime.datetime.now(), string)

def create_image_network_dataloader(dset_type_, data_base_dir_, batch_size_):
    dataset = ImageVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_,
        shuffle=True,
        num_workers=4)
    return dataloader

def create_model_3dautoencoder(voxel_res, embedding_size):
    model = AE_3D(voxel_res, embedding_size)
    return model

def create_model_image_network(embedding_size):
    model = tv.models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, embedding_size)
    return model

def calc_iou_acc(gt, pred, bin_thresh):
    pred[pred < bin_thresh] = 0
    pred[pred >= bin_thresh] = 1
    gt = gt.int()
    pred = pred.int()
    total = 0.0
    for i in xrange(gt.size(0)):
        intersect = (gt[i] * pred[i]).data.nonzero()
        union = (torch.add(gt[i],pred[i])).data.nonzero()
        total += float(len(intersect)) / float(len(union))
    return (float(total) / float(gt.size(0)))

def write_mats(predictions):

    # Make dirs
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_PROB_DIR):
        os.makedirs(OUTPUT_PROB_DIR)
    if not os.path.exists(OUTPUT_BINARY_DIR):
        os.makedirs(OUTPUT_BINARY_DIR)

    # Write mats
    for im_name in predictions:
        
        obj_id = im_name.split("_")[1]
        pred_obj_dir = os.path.join(OUTPUT_PROB_DIR, obj_id)
        if not os.path.exists(pred_obj_dir):
            os.makedirs(pred_obj_dir)
        
        prob_fp = str(os.path.join(pred_obj_dir, "%s.mat" % im_name.split(".")[0]))
        scio.savemat(prob_fp, {"id":im_name, "data":predictions[im_name]})

        for BIN_THRESH in BIN_THRESHES:
            thresh_s = str(BIN_THRESH).replace('.', '_')
            binary_dir = os.path.join(OUTPUT_BINARY_DIR, thresh_s)
            if not os.path.exists(binary_dir):
                os.makedirs(binary_dir)
            obj_binary_dir = os.path.join(binary_dir, obj_id)
            if not os.path.exists(obj_binary_dir):
                os.makedirs(obj_binary_dir)

            # Write mat
            binary_fp = str(os.path.join(obj_binary_dir, "%s.mat" % im_name.split(".")[0]))
            binary_pred = np.copy(predictions[im_name])
            binary_pred[binary_pred < BIN_THRESH] = 0
            binary_pred[binary_pred >= BIN_THRESH] = 1
            scio.savemat(binary_fp, {"id":im_name, "data":binary_pred})
    return

def write_ious(avg_ious):
    fp = os.path.join(OUTPUT_DIR, "iou.txt")
    f = open(fp, 'w')
    f.write("BIN_THRESH: IoU_ACC\n")
    for i in xrange(len(avg_ious)):
        f.write("%s: %s\n" % (BIN_THRESHES[i], avg_ious[i]))
    f.close()

def test_model(model_ae, model_im, test_dataloader, loss_f):

    # Iterate through dataset
    count = 0
    curr_loss = 0.0
    curr_iou = [0.0 for BIN_THRESH in BIN_THRESHES]
    preds = {}
    log_print("\t%i instances" % len(test_dataloader.dataset))
    model_ae.eval()
    model_im.eval()
    for data in test_dataloader:

        log_print("\tIms: %i - %i" % ((BATCH_SIZE*count)+1, BATCH_SIZE*(count+1)))
        count += 1
        # Wrap as pytorch autograd Variable
        ims = data['im']
        voxels = data['voxel']
        if GPU and torch.cuda.is_available():
            ims = ims.cuda()
            voxels = voxels.cuda()
        ims = Variable(ims)
        voxels = Variable(voxels).float()

        # Forward pass
        im_embed = model_im(ims).detach()
        out_voxels = model_ae.module._decode(im_embed).detach()

        # Calculate loss
        loss = loss_f(out_voxels.float(), voxels.float())
        curr_loss += BATCH_SIZE * loss.data[0]

        # Calculate accuracy (IOU accuracy)
        for i in xrange(len(BIN_THRESHES)):
            iou = calc_iou_acc(voxels.clone(), out_voxels.clone(), BIN_THRESHES[i])
            curr_iou[i] += BATCH_SIZE * iou

        # Save out voxels
        out_voxels = out_voxels.cpu()
        for i in xrange(0, out_voxels.size(0)):
            im_name = data['im_name'][i]
            out_vox = out_voxels[i].data.numpy()
            preds[im_name] = out_vox

    # Report results
    num_images = len(test_dataloader.dataset)
    total_loss = float(curr_loss) / float(num_images)
    log_print("\tAverage Loss: %f" % total_loss)
    log_print("\tAverage IoU Acc (by threshold):")
    avg_ious = []
    for i in xrange(len(BIN_THRESHES)):
        BIN_THRESH = BIN_THRESHES[i]
        avg_iou = float(curr_iou[i]) / float(num_images)
        avg_ious.append(avg_iou)
        log_print("\t\t%f:\t%f" % (BIN_THRESH, avg_iou))

    # Finish up
    return preds, avg_ious

def save_model_weights(model, filepath):
    torch.save(model.state_dict(), filepath)

#####################
#    END HELPERS    #
#####################

def main():

    # Create testing DataLoader
    log_print("Beginning script...")
    log_print("Loading testing data...")
    test_dataloader =  \
        create_image_network_dataloader(
            "test",
            DATA_BASE_DIR,
            BATCH_SIZE) 

    # Set up loss func
    loss_f = nn.BCELoss()
    if GPU and torch.cuda.is_available():
        loss_f = loss_f.cuda()

    # Set up model for training
    log_print("Creating model...")
    model_ae = create_model_3dautoencoder(VOXEL_RES, EMBED_SIZE)
    model_im = create_model_image_network(EMBED_SIZE)
    if GPU and torch.cuda.is_available():
        log_print("\tEnabling GPU")
        if MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("\tUsing multiple GPUs: %i" % torch.cuda.device_count())
            model_ae = nn.DataParallel(model_ae)
            model_im = nn.DataParallel(model_im)
        model_ae = model_ae.cuda()
        model_im = model_im.cuda()
    else:
        log_print("\tIgnoring GPU (CPU only)")
    model_ae.load_state_dict(torch.load(IN_AE_WEIGHTS_FP))
    model_im.load_state_dict(torch.load(IN_IM_WEIGHTS_FP))

    # Perform testing and save mats
    log_print("Generating predictions...")
    predictions, avg_ious = test_model(model_ae, model_im, test_dataloader, loss_f)
    log_print("Writing predictions to file...")
    write_mats(predictions)
    write_ious(avg_ious)

    log_print("Script DONE!")

if __name__ == "__main__":
    main()
