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
DATA_BASE_DIR = "../../data/%s" % OBJ_CLASS
IN_WEIGHTS_FP = "../../output/%s/models/test-ae3d.pt" % OBJ_CLASS
OUTPUT_PROB_DIR = "../../output/%s/preds/prob" % OBJ_CLASS
OUTPUT_BINARY_DIR = "../../output/%s/preds/binary" % OBJ_CLASS

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

def write_mats(predictions):

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
    preds = {}
    log_print("\t%i instances" % len(test_dataloader.dataset))
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
        #TODO: Calculate accuracy (IOU accuracy)

        # Save out voxels
        for i in xrange(0, out_voxels.size(0)):
            id_ = data['id'][i]
            out_vox = out_voxels[i].data
            preds[id_] = out_vox

    # Report results
    num_images = len(test_dataloader.dataset)
    total_loss = float(curr_loss) / float(num_images)
    log_print("\tAverage Loss: %f" % (total_loss))

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
    try:
        model.load_state_dict(torch.load(IN_WEIGHTS_FP))
    except:
        log_print("ERROR...weights not loaded")
        pass
    if GPU and torch.cuda.is_available():
        log_print("\tEnabling GPU")
        if MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("\tUsing multiple GPUs: %i" % torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        log_print("\tIgnoring GPU (CPU only)")

    # Perform testing and save mats
    log_print("Generating predictions...")
    predictions = test_model(model, test_dataloader, loss_f)

    log_print("Writing predictions to file...")
    write_mats(predictions)

    log_print("Script DONE!")

if __name__ == "__main__":
    main()
