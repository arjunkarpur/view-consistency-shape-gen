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
from torchvision import transforms
import scipy.io as scio

# Imports from src files
from datasets import ShapeNetVoxelDataset
from models import AE_3D

GPU = True
MULTI_GPU = True
OBJ_CLASS = "CHAIR"
NAME = "chair-ae3d-long"
DATA_BASE_DIR = "../../data/%s" % OBJ_CLASS
IN_WEIGHTS_FP = "../../output/%s/models/%s/%s.pt" % (OBJ_CLASS, NAME, NAME)
OUTPUT_DIR = "../../output/%s/preds/%s" % (OBJ_CLASS, NAME)
OUTPUT_PROB_DIR = "%s/prob" % OUTPUT_DIR
OUTPUT_BINARY_DIR = "%s/binary" % OUTPUT_DIR

VOXEL_RES = 20
EMBED_SIZE = 64
BATCH_SIZE = 32
BIN_THRESH = 0.5

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
    print "[%s]\t %s" % (datetime.datetime.now(), string)

def create_shapenet_voxel_dataloader(dset_type_, data_base_dir_, batch_size_):
    dataset = ShapeNetVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_,
        transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_,
        shuffle=True,
        num_workers=4)
    return dataloader

def create_model(voxel_res, embedding_size):
    model = AE_3D(voxel_res, embedding_size)
    return model

def calc_iou_acc(gt, pred, bin_thresh):
    pred[pred < bin_thresh] = 0
    pred[pred >= bin_thresh] = 1
    gt = gt.int()
    pred = pred.int()
    intersect = (gt * pred).data.nonzero()
    union = (torch.add(gt,pred)).data.nonzero()
    return float(len(intersect)) / float(len(union))

def write_mats(predictions):

    # Make dirs
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_PROB_DIR):
        os.makedirs(OUTPUT_PROB_DIR)
    if not os.path.exists(OUTPUT_BINARY_DIR):
        os.makedirs(OUTPUT_BINARY_DIR)

    # Write mats
    for id_ in predictions:
        prob_fp = str(os.path.join(OUTPUT_PROB_DIR, "%s.mat" % id_))
        scio.savemat(prob_fp, {"id":id_, "data":predictions[id_]})
        binary_fp = str(os.path.join(OUTPUT_BINARY_DIR, "%s.mat" % id_))
        binary_pred = predictions[id_]
        binary_pred[binary_pred < BIN_THRESH] = 0
        binary_pred[binary_pred >= BIN_THRESH] = 1
        scio.savemat(binary_fp, {"id":id_, "data":binary_pred})
    return

def test_model(model, test_dataloader, loss_f):

    # Iterate through dataset
    curr_loss = 0.0
    curr_iou = 0.0
    preds = {}
    log_print("\t%i instances" % len(test_dataloader.dataset))
    model.eval()
    for data in test_dataloader:

        # Wrap as pytorch autograd Variable
        voxels = data['data']
        if GPU and torch.cuda.is_available():
            voxels = voxels.cuda()
        voxels = Variable(voxels).float()

        # Forward pass
        out_voxels = model(voxels)

        # Calculate loss
        loss = loss_f(out_voxels.float(), voxels.float())
        curr_loss += BATCH_SIZE * loss.data[0]

        # Calculate accuracy (IOU accuracy)
        iou = calc_iou_acc(voxels, out_voxels, BIN_THRES)
        curr_iou += config.BATCH_SIZE * iou

        # Save out voxels
        out_voxels = out_voxels.cpu()
        for i in xrange(0, out_voxels.size(0)):
            id_ = data['id'][i]
            out_vox = out_voxels[i].data.numpy()
            preds[id_] = out_vox

    # Report results
    num_images = len(test_dataloader.dataset)
    total_loss = float(curr_loss) / float(num_images)
    total_iou = float(curr_iou) / float(num_images)
    log_print("\tAverage Loss: %f" % total_loss)
    log_print("\tAverage IoU Acc: %f" % total_iou)

    # Finish up
    return preds 

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
        create_shapenet_voxel_dataloader(
            "test",
            DATA_BASE_DIR,
            BATCH_SIZE) 

    # Set up loss func
    loss_f = nn.BCELoss()
    if GPU and torch.cuda.is_available():
        loss_f = loss_f.cuda()

    # Set up model for training
    log_print("Creating model...")
    model = create_model(VOXEL_RES, EMBED_SIZE)
    if GPU and torch.cuda.is_available():
        log_print("\tEnabling GPU")
        if MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("\tUsing multiple GPUs: %i" % torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        log_print("\tIgnoring GPU (CPU only)")
    model.load_state_dict(torch.load(IN_WEIGHTS_FP))

    # Perform testing and save mats
    log_print("Generating predictions...")
    predictions = test_model(model, test_dataloader, loss_f)
    log_print("Writing predictions to file...")
    write_mats(predictions)

    log_print("Script DONE!")

if __name__ == "__main__":
    main()
