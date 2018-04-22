import os
import sys
import math
import config
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision as tv
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# Imports from src files
from datasets import ShapeNetVoxelDataset
from models import AE_3D

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
    print "[%s]\t %s" % (datetime.datetime.now(), string)

def create_shapenet_voxel_dataloader(dset_type_, data_base_dir_, batch_size_):
    dataset = ShapeNetVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_,
        transform=tv.transforms.Compose([tv.transforms.ToTensor()]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_,
        shuffle=True,
        num_workers=4)
    return dataloader

def create_image_network_dataloader(dset_type_, data_base_dir_, batch_size_):
    #TODO
    """
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
    """
    return

def create_model_3d_autoencoder(voxel_res, embedding_size):
    model = AE_3D(voxel_res, embedding_size)
    return model

def create_model_image_network(embedding_size):

    # Create resnet50 layer
    model = tv.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, embedding_size)
    nn.init.normal(model.fc.weight, mean=0, std=0.01)

    # Fix layers up to and including layer 2
    child_counter = 0
    for child in model.children():
        if child_counter < 6:
            for param in child.parameters():
                param.requires_grad = False
        else:
            break
        child_counter += 1

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
        total += (float(len(intersect)) / float(len(union)))
    return (float(total) / float(gt.size(0)))

def train_model(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):

    init_time = time.time()
    best_loss = float('inf')
    best_weights = model.state_dict().copy()
    best_epoch = -1

    for epoch in xrange(epochs):
        lr = []
        for param_group in optimizer.param_groups:
            lr += [str(param_group['lr'])]

        log_print("Epoch %i/%i: %i batches of %i images each. LR: [%s]" % (epoch+1, epochs, len(train_dataloader.dataset)/config.BATCH_SIZE, config.BATCH_SIZE, ','.join(lr)))

        for phase in ["train", "val"]:
            if phase == "train":
                if explorer is not None:
                    explorer.step()
                model.eval()
                dataloader = train_dataloader
            else:
                model.train()
                dataloader = val_dataloader

            # Iterate over dataset
            epoch_loss = 0.0
            epoch_iou = 0.0
            curr_loss = 0.0
            curr_iou = 0.0
            print_interval = 20
            epoch_checkpoint = config.WEIGHTS_CHECKPOINT
            batch_count = 0

            # Iterate through dataset
            for data in dataloader:

                # Wrap as pytorch autograd Variable
                voxels = data['data']
                if config.GPU and torch.cuda.is_available():
                    voxels = voxels.cuda()
                voxels = Variable(voxels).float()

                # Forward pass
                optimizer.zero_grad()
                out_voxels = model(voxels)

                # Calculate loss
                loss = loss_f(out_voxels.float(), voxels.float())
                curr_loss += config.BATCH_SIZE * loss.data[0]

                # Backward pass (if train)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # Calculate accuracy (IOU accuracy)
                iou = calc_iou_acc(voxels, out_voxels, config.IOU_THRESH)
                curr_iou += config.BATCH_SIZE * iou

                # Output
                if batch_count % print_interval == 0 and batch_count != 0:
                    epoch_loss += curr_loss
                    epoch_iou += curr_iou
                    if phase == "train":
                        curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
                        curr_iou = float(curr_iou) / float(print_interval*config.BATCH_SIZE)
                        log_print("\tBatches %i-%i -\tAvg Loss: %f ,  Avg IoU: %f" % (batch_count-print_interval+1, batch_count, curr_loss, curr_iou))
                    curr_loss = 0.0
                    curr_iou
                batch_count += 1
      
            # Report epoch results
            num_images = len(dataloader.dataset)
            epoch_loss = float(epoch_loss+curr_loss) / float(num_images)
            epoch_iou = float(epoch_iou+curr_iou) / float(num_images)
            log_print("\tEPOCH %i [%s] - Avg Loss: %f , Avg IoU: %f" % (epoch+1, phase, epoch_loss, epoch_iou))

            # Save best model weights from epoch
            err_improvement = best_loss - epoch_loss 
            if phase == "val":
                if err_improvement >= 0 or epoch == 0:
                    log_print("\tBEST NEW EPOCH: %i" % (epoch+1))
                    best_loss = epoch_loss
                    best_weights = model.state_dict().copy()
                    best_epoch = epoch
                else:
                    log_print("\tCurrent best epoch: %i, %f loss" % (best_epoch, best_loss))

            # Checkpoint epoch weights
            if (phase == "val") and (epoch % epoch_checkpoint == 0) and (epoch != 0):
                log_print("\tCheckpointing weights for epoch %i" % (epoch + 1))
                save_model_weights(model, "%s_%i" % (config.RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch+1, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_weights)
    return model


def train_model_im_network(model_ae, model_im, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):
    #TODO
    return model_im

def save_model_weights(model, name):
    dir_ = config.OUT_WEIGHTS_DIR
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fp = os.path.join(dir_, "%s.pt" % name)
    torch.save(model.state_dict().copy(), fp)

def try_load_weights(model, fp):
    try:
        model.load_state_dict(torch.load(fp))
        return True
    except:
        return False

def train_autoencoder():
    # Create training DataLoader
    log_print("Loading training data...")
    train_dataloader =  \
        create_shapenet_voxel_dataloader(
            "train",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE) 
    val_dataloader = \
        create_shapenet_voxel_dataloader(
            "val",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE)

    # Set up model for training
    log_print("Creating model...")
    model = create_model(config.VOXEL_RES, config.EMBED_SIZE)
    load_success = False
    if config.LOAD_WEIGHTS is not None:
        load_success = try_load_weights(model, config.LOAD_WEIGHTS)
    if config.GPU and torch.cuda.is_available():
        log_print("\tEnabling GPU")
        if config.MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("\tUsing multiple GPUs: %i" % torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        log_print("\tIgnoring GPU (CPU only)")
    if (config.LOAD_WEIGHTS is not None) and (load_success is False):
        if not try_load_weights(model, config.LOAD_WEIGHTS):
            log_print("FAILED TO LOAD PRE-TRAINED WEIGHTS. EXITING")
            sys.exit(-1)

    # Set up loss and optimizer
    loss_f = nn.BCELoss()
    if config.GPU and torch.cuda.is_available():
        loss_f = loss_f.cuda()
    """
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM)
    """
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=config.LEARNING_RATE)
    if config.STEP_SIZE is not None:
        explorer = lr_scheduler.StepLR(
            optimizer, 
            step_size=config.STEP_SIZE,
            gamma=config.GAMMA)
    elif config.LR_STEPS is not None:
        explorer = lr_scheduler.MultiStepLR(
            optimizer,
            config.LR_STEPS,
            gamma=config.GAMMA)
    else:
        explorer = None

    # Perform training
    log_print("~~~~~Starting training~~~~~")
    model = train_model(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, config.EPOCHS)
    log_print("~~~~~Training finished~~~~~")
  
    # Save model weights
    out_fp = os.path.join(config.OUT_WEIGHTS_DIR, "%s.pt" % config.RUN_NAME)
    log_print("Saving model weights to %s..." % out_fp)
    save_model_weights(model, config.RUN_NAME)
    return

def train_image_network():
    log_print("Loading training data...")
    train_dataloader = \
        create_image_network_dataloader(
            "train",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE)
    val_dataloader = \
        create_image_network_dataloader(
            "val",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE)

    log_print("Creating image network and 3d-autoencoder models...")
    model_ae = create_model_3d_autoencoder(config.VOXEL_RES, config.EMBED_SIZE)
    model_im = create_model_image_network(config.EMBED_SIZE)
    try_load_weights(model_ae, config.AE3D_LOAD_WEIGHTS)
    if config.GPU and torch.cuda.is_available():
        log_print("\tEnabling GPU")
        if config.MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("\tUsing multiple GPUs: %i" % torch.cuda.device_count())
            model_ae = nn.DataParallel(model_ae)
            model_im = nn.DataParallel(model_im)
        model_ae = model_ae.cuda()
        model_im = model_im.cuda()
    else:
        log_print("\t Ignoring GPU (CPU only)")


    # Set up loss and optimizer
    loss_f = nn.MSELoss()
    if config.GPU and torch.cuda.is_available():
        loss_f = loss_f.cuda()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model_im.parameters()),
        lr=config.IM_LEARNING_RATE,
        momentum=config.IM_MOMENTUM)
    explorer = None

    # Perform training
    log_print("~~~~~Start training~~~~~")
    model_im = train_model_im_network(model_ae, model_im, train_dataloader, val_dataloader, loss_f, optimizer, explorer, config.EPOCHS)
    log_print("~~~~~Training finished~~~~~")

    # Save model weights
    log_print("Saving model weights")
    save_model_weights(model_im, config.IM_RUN_NAME)
    return

def train_joint():
    return

#####################
#    END HELPERS    #
#####################

def main():

    # Redirect output to log file
    #sys.stdout = open(config.OUT_LOG_FP, 'w')
    #sys.stderr = sys.stdout
    log_print("Beginning script...")

    # Print beginning debug info
    log_print("Printing config file...")
    config.PRINT_CONFIG()

    # Run 3 step training process
    log_print("BEGINNING PART 1: train 3D autoencoder")
    #train_autoencoder()
    log_print("FINISHING PART 1") 

    log_print("BEGINNING PART 2: train image network")
    train_image_network()
    log_print("FINISHING PART 1") 

    log_print("BEGINNING PART 3: joint training")
    train_joint()
    log_print("FINISHING PART 3") 

    # Finished
    log_print("Script DONE!")

if __name__ == "__main__":
    main()
