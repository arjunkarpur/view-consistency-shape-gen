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
from datasets import ShapeNetVoxelDataset, RenderingVoxelDataset
from models import AE_3D

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
    print "[%s]\t %s" % (datetime.datetime.now(), string)
    sys.stdout.flush()
    os.fsync(sys.stdout.fileno())
    return

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
    dataset = RenderingVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_,
        shuffle=True,
        num_workers=4)
    return dataloader

def create_real_image_dataloader(dset_type_, data_base_dir_, batch_size_):
    dataset = RealVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size_,
        shuffle=True,
        num_workers=4)
    return dataloader

def create_model_3d_autoencoder(voxel_res, embedding_size):
    model = AE_3D(voxel_res, embedding_size)
    return model

def create_model_image_network(embedding_size):

    # Create resnet50 layer
    model = tv.models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, embedding_size)
    nn.init.xavier_uniform(model.fc.weight)

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

def train_model_ae3d(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):

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
            print_interval = config.AE_PRINT_INTERVAL
            epoch_checkpoint = config.AE_WEIGHTS_CHECKPOINT
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
                voxels = voxels.detach()
                out_voxels = out_voxels.detach()
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
                    curr_iou = 0.0
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
                save_model_weights(model, "%s_%i" % (config.AE_RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch+1, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_weights)
    return model


def train_model_im_network(model_ae, model_im, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):

    # Freeze all 3d autoencoder layers
    for param in model_ae.parameters():
        param.requires_grad = False
        
    init_time = time.time()
    best_loss = float('inf')
    best_weights = model_im.state_dict().copy()
    best_epoch = -1
    model_ae.eval()

    for epoch in xrange(epochs):
        log_print("Epoch %i/%i: %i batches of %i images each" % (epoch+1, epochs, len(train_dataloader.dataset)/config.BATCH_SIZE, config.BATCH_SIZE))

        for phase in ["train", "val"]:
            if phase == "train":
                if explorer is not None:
                    explorer.step()
                model_im.train()
                dataloader = train_dataloader
            else:
                model_im.eval()
                dataloader = val_dataloader

            # Iterate over dataset
            epoch_loss = 0.0
            curr_loss = 0.0
            print_interval = config.IM_PRINT_INTERVAL
            epoch_checkpoint = config.IM_WEIGHTS_CHECKPOINT
            batch_count = 0

            for data in dataloader:

                # Wrap in pytorch autograd Variable
                ims, voxels = data['im'], data['voxel']
                if config.GPU and torch.cuda.is_available():
                    ims = ims.cuda()
                    voxels = voxels.cuda()
                ims = Variable(ims).float()
                voxels = Variable(voxels).float()

                # Forward passes + calc loss
                optimizer.zero_grad()
                im_embed = model_im(ims)
                voxel_embed = model_ae.module._encode(voxels)
                loss = loss_f(im_embed, voxel_embed)
                curr_loss += config.BATCH_SIZE * loss.data[0]

                # Backprop and cleanup
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # Output
                if batch_count % print_interval == 0 and batch_count != 0:
                    epoch_loss += curr_loss
                    if phase == "train":
                        curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
                        log_print("\tBatches %i-%i -\tAvg Loss: %f" % (batch_count-print_interval+1, batch_count, curr_loss))
                    curr_loss = 0.0
                batch_count += 1

            # Report epoch results
            num_images = len(dataloader.dataset)
            epoch_loss = float(epoch_loss + curr_loss) / float(num_images)
            log_print("\tEPOCH %i [%s] - Avg Loss: %f" % (epoch+1, phase, epoch_loss))
            
            # Save best model weights from epoch
            err_improvement = best_loss - epoch_loss 
            if phase == "val":
                if err_improvement >= 0 or epoch == 0:
                    log_print("\tBEST NEW EPOCH: %i" % (epoch+1))
                    best_loss = epoch_loss
                    best_weights = model_im.state_dict().copy()
                    best_epoch = epoch
                else:
                    log_print("\tCurrent best epoch: %i, %f loss" % (best_epoch, best_loss))

            # Checkpoint epoch weights
            if (phase == "val") and (epoch % epoch_checkpoint == 0) and (epoch != 0):
                log_print("\tCheckpointing weights for epoch %i" % (epoch + 1))
                save_model_weights(model_im, "%s_%i" % (config.IM_RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch+1, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    model_im.load_state_dict(best_weights)
    return model_im
    

def train_model_joint(model_ae, model_im, train_dataloader, val_dataloader, loss_ae_f, loss_im_f, optimizer, explorer, epochs):

    # Init metrics
    init_time = time.time()
    best_loss = float('inf')
    best_weights_ae = model_ae.state_dict().copy()
    best_weights_im = model_im.state_dict().copy()
    best_epoch = -1

    for epoch in xrange(epochs):
        log_print("Epoch %i/%i: %i batches of %i images each" % (epoch+1, epochs, len(train_dataloader.dataset)/config.BATCH_SIZE, config.BATCH_SIZE))

        for phase in ["train", "val"]:
            if phase == "train":
                if explorer is not None:
                    explorer.step()
                model_ae.train()
                model_im.train()
                dataloader = train_dataloader
            else:
                model_ae.eval()
                model_im.eval()

            # Iterate over dataset
            epoch_loss_ae = epoch_loss_im = 0.0
            curr_loss_ae = curr_loss_im = 0.0
            epoch_iou = 0.0
            curr_iou = 0.0
            print_interval = config.JOINT_PRINT_INTERVAL
            epoch_checkpoint = config.JOINT_WEIGHTS_CHECKPOINT
            batch_count = 0

            for data in dataloader:
                
                # Wrap in pytorch autograd vars
                ims, voxels = data['im'], data['voxel']
                if config.GPU and torch.cuda.is_available():
                    ims = ims.cuda()
                    voxels = voxels.cuda()
                ims = Variable(ims).float()
                voxels = Variable(voxels).float()

                # Forward pass
                optimizer.zero_grad()
                im_embed = model_im(ims)
                voxel_embed = model_ae.module._encode(voxels)
                out_voxels = model_ae.module._decode(voxel_embed)

                # Calc loss
                loss_ae = config.JOINT_LAMBDA_AE * loss_ae_f(out_voxels, voxels)
                #loss_im = config.JOINT_LAMBDA_IM * loss_im_f(im_embed, voxel_embed)
                loss_im = config.JOINT_LAMBDA_IM * (
                    torch.sum((im_embed - voxel_embed)**2) / im_embed.data.nelement())
                total_loss = loss_ae + loss_im
                curr_loss_ae += config.BATCH_SIZE * loss_ae.data[0]
                curr_loss_im += config.BATCH_SIZE * loss_im.data[0]

                # Backprop and cleanup
                if phase == "train":
                    total_loss.backward()
                    optimizer.step()

                # Calculate accuracy (IOU)
                voxels = voxels.detach()
                out_voxels = out_voxels.detach()
                iou = calc_iou_acc(voxels, out_voxels, config.IOU_THRESH)
                curr_iou += config.BATCH_SIZE * iou

                # Output
                if batch_count % print_interval == 0 and batch_count != 0:
                    epoch_loss_ae += curr_loss_ae
                    epoch_loss_im += curr_loss_im
                    epoch_iou += curr_iou
                    if phase == "train":
                        curr_loss_ae = float(curr_loss_ae) / float(print_interval*config.BATCH_SIZE)
                        curr_loss_im = float(curr_loss_im) / float(print_interval*config.BATCH_SIZE)
                        curr_iou = float(curr_iou) / float(print_interval*config.BATCH_SIZE)
                        log_print("\tBatches %i-%i -\tAvg Loss: %f | Avg AE Loss: %f , Avg Im Loss: %f , Avg IoU: %f" % (batch_count-print_interval+1, batch_count, curr_loss_ae+curr_loss_im, curr_loss_ae, curr_loss_im, curr_iou))
                    curr_loss_ae = curr_loss_im = curr_iou = 0.0
                batch_count += 1

            # Report epoch results
            num_images = len(dataloader.dataset)
            epoch_loss_ae = float(epoch_loss_ae + curr_loss_ae) / float(num_images)
            epoch_loss_im = float(epoch_loss_im + curr_loss_im) / float(num_images)
            total_epoch_loss = epoch_loss_ae + epoch_loss_im
            epoch_iou = float(epoch_iou + curr_iou) / float(num_images)
            log_print("\tEPOCH %i [%s] - Avg Loss: %f | Avg AE Loss: %f , Avg Im Loss: %f , Avg IoU: %f" % (epoch+1, phase, total_epoch_loss, epoch_loss_ae, epoch_loss_im, epoch_iou))

            # Save best model weights
            err_improvement = best_loss - total_epoch_loss
            if phase == "val":
                if err_improvement >= 0 or epoch == 0:
                    log_print("\tBEST NEW EPOCH: %i" % (epoch+1))
                    best_loss = total_epoch_loss
                    best_weights_ae = model_ae.state_dict().copy()
                    best_weights_im = model_im.state_dict().copy()
                    best_epoch = epoch
                else:
                    log_print("\tCurrent best epoch: %i, %f loss" % (best_epoch, best_loss))

            # Checkpoint epoch weights
            if (phase == "val") and (epoch % epoch_checkpoint == 0):
                log_print("\tCheckpoint weights for epoch %i" % (epoch + 1))
                save_model_weights(model_ae, "%s_ae3d_%i" % (config.JOINT_RUN_NAME, epoch))
                save_model_weights(model_im, "%s_im_%i" % (config.JOINT_RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch+1, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    model_ae.load_state_dict(best_weights_ae)
    model_im.load_state_dict(best_weights_im)
    return (model_ae, model_im)

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
    model = create_model_3d_autoencoder(config.VOXEL_RES, config.EMBED_SIZE)
    load_success = False
    if config.AE_INIT_WEIGHTS is not None:
        load_success = try_load_weights(model, config.AE_INIT_WEIGHTS)
    if config.GPU and torch.cuda.is_available():
        log_print("\tEnabling GPU")
        if config.MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("\tUsing multiple GPUs: %i" % torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        log_print("\tIgnoring GPU (CPU only)")
    if (config.AE_INIT_WEIGHTS is not None) and (load_success is False):
        if not try_load_weights(model, config.AE_INIT_WEIGHTS):
            log_print("FAILED TO LOAD PRE-TRAINED WEIGHTS. EXITING")

    # Set up loss and optimizer
    loss_f = nn.BCELoss()
    if config.GPU and torch.cuda.is_available():
        loss_f = loss_f.cuda()
    optimizer = optim.Adadelta(
        model.parameters(),
        lr=config.AE_LEARNING_RATE)
    explorer = None

    # Perform training
    log_print("~~~~~Starting training~~~~~")
    model = train_model_ae3d(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, config.AE_EPOCHS)
    log_print("~~~~~Training finished~~~~~")
  
    # Save model weights
    out_fp = os.path.join(config.OUT_WEIGHTS_DIR, "%s.pt" % config.AE_RUN_NAME)
    log_print("Saving model weights to %s..." % out_fp)
    save_model_weights(model, config.AE_RUN_NAME)
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
    load_success_ae = try_load_weights(model_ae, config.IM_AE3D_LOAD_WEIGHTS)
    if config.IM_INIT_WEIGHTS is not None:
        load_success_im = try_load_weights(model_im, config.IM_INIT_WEIGHTS)
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
    if not load_success_ae:
        if not try_load_weights(model_ae, config.IM_AE3D_LOAD_WEIGHTS):
            log_print("COULDN'T LOAD AUTOENCODER WEIGHTS")
    if config.IM_INIT_WEIGHTS is not None and not load_success_im:
        if not try_load_weights(model_im, config.IM_INIT_WEIGHTS):
            log_print("COULDN'T LOAD IMAGE NETWORK INIT WEIGHTS")

    # Set up loss and optimizer
    loss_f = nn.MSELoss()
    if config.GPU and torch.cuda.is_available():
        loss_f = loss_f.cuda()
    """
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model_im.parameters()),
        lr=config.IM_LEARNING_RATE,
        momentum=config.IM_MOMENTUM)
    """
    optimizer = optim.Adadelta(
        filter(lambda p: p.requires_grad, model_im.parameters()),
        lr=config.IM_LEARNING_RATE)
    explorer = None

    # Perform training
    log_print("~~~~~Start training~~~~~")
    model_im = train_model_im_network(model_ae, model_im, train_dataloader, val_dataloader, loss_f, optimizer, explorer, config.IM_EPOCHS)
    log_print("~~~~~Training finished~~~~~")

    # Save model weights
    log_print("Saving model weights")
    save_model_weights(model_im, config.IM_RUN_NAME)
    return

def train_joint():

    # Load train + val data
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

    # Create models
    log_print("Creating image network and 3d-autoencoder models...")
    model_ae = create_model_3d_autoencoder(config.VOXEL_RES, config.EMBED_SIZE)
    model_im = create_model_image_network(config.EMBED_SIZE)
    load_success_ae = try_load_weights(model_ae, config.JOINT_AE3D_LOAD_WEIGHTS)
    load_success_im = try_load_weights(model_im, config.JOINT_IM_LOAD_WEIGHTS)
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
    if not load_success_ae:
        if not (try_load_weights(model_ae, config.JOINT_AE3D_LOAD_WEIGHTS)):
            log_print("FAILED TO LOAD AE3D WEIGHTS")
    if not load_success_im:
        if not try_load_weights(model_im, config.JOINT_IM_LOAD_WEIGHTS):
            log_print("FAILED TO LOAD IMAGE NETWORK WEIGHTS")

    # Set up loss and optimizer
    loss_ae = nn.BCELoss()
    loss_im = nn.MSELoss()
    if config.GPU and torch.cuda.is_available():
        loss_ae = loss_ae.cuda()
        loss_im = loss_im.cuda()
    params = list(model_ae.parameters()) + \
        list(filter(lambda p: p.requires_grad, model_im.parameters()))
    optimizer = optim.SGD(
        params,
        lr=config.JOINT_LEARNING_RATE,
        momentum=config.JOINT_MOMENTUM)
    explorer = None

    # Perform training
    log_print("~~~~~Start training~~~~~")
    model_ae, model_im = train_model_joint(model_ae, model_im, train_dataloader, val_dataloader, loss_ae, loss_im, optimizer, explorer, config.JOINT_EPOCHS)
    log_print("~~~~~Training finished~~~~~")

    # Save model weights
    log_print("Saving model weights")
    save_model_weights(model_ae, "%s_ae3d" % config.JOINT_RUN_NAME)
    save_model_weights(model_im, "%s_im" % config.JOINT_RUN_NAME)

    return

#####################
#    END HELPERS    #
#####################

def main():

    # Redirect output to log file
    sys.stdout = open(config.OUT_LOG_FP, 'w')
    sys.stderr = sys.stdout
    log_print("Beginning script...")

    # Print beginning debug info
    log_print("Printing config file...")
    config.PRINT_CONFIG()

    # Run 3 step training process
    log_print("***BEGINNING PART 1: train 3D autoencoder")
    train_autoencoder()
    log_print("***FINISHED PART 1") 

    log_print("***BEGINNING PART 2: train image network")
    train_image_network()
    log_print("***FINISHED PART 1") 

    log_print("***BEGINNING PART 3: joint training")
    train_joint()
    log_print("***FINISHED PART 3") 

    # Finished
    log_print("Script DONE!")

if __name__ == "__main__":
    main()
