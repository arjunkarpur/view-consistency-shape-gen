import os
import sys
import copy
import time
import math
import torch
import config
import random
import datetime
import numpy as np
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, sampler

# Imports from src files
from datasets import VoxelDataset, ImageVoxelDataset, FusionDataset
from models import AE_3D
import optim_latent

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
    print "[%s]\t %s" % (datetime.datetime.now(), string)
    sys.stdout.flush()
    os.fsync(sys.stdout.fileno())
    return

###########################
#   DATA

def create_dataloader_from_dataset(dataset_, batch_size_, subset_size_=None):

    # Subset of dataset
    if subset_size_ is not None:
        indices = range(len(dataset_))
        indices = random.sample(indices, min(subset_size_, len(dataset_)))
        sampler_ = sampler.SubsetRandomSampler(indices)
        shuffle_ = False
    else:
        sampler_ = None
        shuffle_ = True

    dataloader = DataLoader(
        dataset_,
        sampler=sampler_,
        batch_size=batch_size_,
        shuffle=shuffle_,
        num_workers=4)
    return dataloader

def create_shapenet_voxel_dataloader(dset_type_, data_base_dir_, batch_size_, subset_size_=None):
    dataset_ = VoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_)
    return create_dataloader_from_dataset(dataset_, batch_size_, subset_size_)

def create_image_network_dataloader(dset_type_, data_base_dir_, batch_size_, subset_size_=None):
    dataset_ = ImageVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=data_base_dir_)
    return create_dataloader_from_dataset(dataset_, batch_size_, subset_size_)

def create_fusion_dataloader(dataset_, batch_size_, src_sample_mult=None):
    
    # If no subsamplig required, return normal dataloader
    if src_sample_mult is None:
        return create_dataloader_from_dataset(dataset_, batch_size_, subset_size_=None)

    # Determine indices
    indices_src = range(len(dataset_.src_dataset))
    indices_target = [i+len(indices_src) for i in xrange(len(dataset_.target_dataset))]
    num_src = int(min(src_sample_mult * len(indices_target), len(indices_src)))
    indices_src = random.sample(indices_src, num_src)
    indices_both = indices_src + indices_target

    # Create dataloader
    sampler_ = sampler.SubsetRandomSampler(indices_both)
    dataloader = DataLoader(
        dataset_,
        sampler=sampler_,
        batch_size=batch_size_,
        shuffle=False,
        num_workers=4)
    return dataloader

def create_view_consistency_dataloader(dset_type_, src_data_base_dir_, target_data_base_dir_, batch_size_, src_sample_mult=None):
    src_dataset = ImageVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=src_data_base_dir_)
    target_dataset = ImageVoxelDataset(
        dset_type=dset_type_,
        data_base_dir=target_data_base_dir_)
    dataset_ = FusionDataset(
        src_dataset,
        target_dataset)
    return create_fusion_dataloader(dataset_, batch_size_, src_sample_mult)
    
######################
#   MODELS

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

def save_model_weights(model, name):
    dir_ = config.OUT_WEIGHTS_DIR
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    fp = os.path.join(dir_, "%s.pt" % name)
    torch.save(model.state_dict(), fp)

def try_load_weights(model, fp):
    try:
        model.load_state_dict(torch.load(fp))
        return True
    except:
        return False

############################
#   TRAINING HELPERS

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

def train_model_ae3d(model, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):

    init_time = time.time()
    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    best_epoch = 0

    for epoch in xrange(1, epochs+1):
        lr = []
        for param_group in optimizer.param_groups:
            lr += [str(param_group['lr'])]

        log_print("Epoch %i/%i: LR: [%s]" % (epoch, epochs, ','.join(lr)))

        for phase in ["train", "val"]:
            if phase == "train":
                if explorer is not None:
                    explorer.step()
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader
            if dataloader.sampler is None:
                num_images = len(dataloader.dataset)
            else:
                num_images = len(dataloader.sampler)
            log_print("\t[%s] - %i ims, %i batches" % (phase, num_images, num_images/config.BATCH_SIZE))

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
                curr_loss += voxels.size(0) * loss.item()

                # Backward pass (if train)
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                else:
                    # Switch back to train before any saving
                    model.train()

                # Calculate accuracy (IOU accuracy)
                voxels = voxels.detach()
                out_voxels = out_voxels.detach()
                iou = calc_iou_acc(voxels, out_voxels, config.IOU_THRESH)
                curr_iou += voxels.size(0) * iou

                # Output
                if batch_count % print_interval == 0 and batch_count != 0:
                    epoch_loss += curr_loss
                    epoch_iou += curr_iou
                    if phase == "train":
                        curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
                        curr_iou = float(curr_iou) / float(print_interval*config.BATCH_SIZE)
                        if batch_count < 100:
                            log_print("\tBatches %i-%i -\t\tAvg Loss: %f ,  Avg IoU: %f" % (batch_count-print_interval+1, batch_count, curr_loss, curr_iou))
                        else:
                            log_print("\tBatches %i-%i -\tAvg Loss: %f ,  Avg IoU: %f" % (batch_count-print_interval+1, batch_count, curr_loss, curr_iou))
                    curr_loss = 0.0
                    curr_iou = 0.0
                batch_count += 1
      
            # Report epoch results
            epoch_loss = float(epoch_loss+curr_loss) / float(num_images)
            epoch_iou = float(epoch_iou+curr_iou) / float(num_images)
            log_print("\tEPOCH %i [%s] - Avg Loss: %f , Avg IoU: %f" % (epoch, phase, epoch_loss, epoch_iou))

            # Save best model weights from epoch
            err_improvement = best_loss - epoch_loss 
            if phase == "val":
                if err_improvement >= 0:
                    if not model.module.training:
                        log_print("ERROR: should be in training mode before saving")
                        model.train()
                    log_print("\tBEST NEW EPOCH: %i" % epoch)
                    best_loss = epoch_loss
                    best_model.load_state_dict(model.state_dict())
                    best_epoch = epoch
                    save_model_weights(best_model, config.AE_RUN_NAME)
                else:
                    log_print("\tCurrent best epoch: %i, %f loss" % (best_epoch, best_loss))

            # Checkpoint epoch weights
            if (phase == "val") and (epoch % epoch_checkpoint == 0):
                log_print("\tCheckpointing weights for epoch %i" % epoch)
                save_model_weights(model, "%s_%i" % (config.AE_RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    del model
    return best_model

def train_model_im_network(model_ae, model_im, train_dataloader, val_dataloader, loss_f, optimizer, explorer, epochs):

    # Freeze all 3d autoencoder layers
    for param in model_ae.parameters():
        param.requires_grad = False
        
    init_time = time.time()
    best_loss = float('inf')
    best_model = copy.deepcopy(model_im)
    best_epoch = 0
    model_ae.eval()

    for epoch in xrange(1, epochs+1):
        log_print("Epoch %i/%i:" % (epoch, epochs))

        for phase in ["train", "val"]:
            if phase == "train":
                if explorer is not None:
                    explorer.step()
                model_im.train()
                dataloader = train_dataloader
            else:
                model_im.eval()
                dataloader = val_dataloader
            if dataloader.sampler is None:
                num_images = len(dataloader.dataset)
            else:
                num_images = len(dataloader.sampler)
            log_print("\t[%s] - %i ims, %i batches" % (phase, num_images, num_images/config.BATCH_SIZE))

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
                curr_loss += voxels.size(0) * loss.item()

                # Backprop and cleanup
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                else:
                    model_im.train()

                # Output
                if batch_count % print_interval == 0 and batch_count != 0:
                    epoch_loss += curr_loss
                    if phase == "train":
                        curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
                        log_print("\tBatches %i-%i -\tAvg Loss: %f" % (batch_count-print_interval+1, batch_count, curr_loss))
                    curr_loss = 0.0
                batch_count += 1

            # Report epoch results
            epoch_loss = float(epoch_loss + curr_loss) / float(num_images)
            log_print("\tEPOCH %i [%s] - Avg Loss: %f" % (epoch, phase, epoch_loss))
            
            # Save best model weights from epoch
            err_improvement = best_loss - epoch_loss 
            if phase == "val":
                if err_improvement >= 0:
                    if not model_im.module.training:
                        log_print("ERROR! Should be in training mode before saving")
                        model.train()
                    log_print("\tBEST NEW EPOCH: %i" % epoch)
                    best_loss = epoch_loss
                    best_model.load_state_dict(model_im.state_dict())
                    best_epoch = epoch
                    save_model_weights(best_model, config.IM_RUN_NAME)
                else:
                    log_print("\tCurrent best epoch: %i, %f loss" % (best_epoch, best_loss))

            # Checkpoint epoch weights
            if (phase == "val") and (epoch % epoch_checkpoint == 0):
                log_print("\tCheckpointing weights for epoch %i" % epoch)
                save_model_weights(model_im, "%s_%i" % (config.IM_RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    del model_ae, model_im
    return best_model

def train_model_joint(model_ae, model_im, train_dataloader, val_dataloader, loss_ae_f, loss_im_f, optimizer, explorer, epochs):

    # Init metrics
    init_time = time.time()
    best_loss = float('inf')
    best_model_ae = copy.deepcopy(model_ae)
    best_model_im = copy.deepcopy(model_im)
    best_epoch = 0
    lambda_ae = 1.0

    # Auto-calc scaling lambda
    if epochs > 0:
        model_ae.eval()
        model_im.eval()
        loss_ae = 0.0
        loss_im = 0.0
        batches_sample = 100
        batch_count = 0
        for data in train_dataloader:
            if batch_count == batches_sample:
                break

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
            loss_ae += loss_ae_f(out_voxels, voxels).item()
            loss_im += (torch.sum((im_embed - voxel_embed)**2) / im_embed.data.nelement()).item()
            batch_count += 1

        lambda_ae = (float(loss_im) / float(loss_ae)) * 0.5
        del loss_ae, loss_im
        model_ae.train()
        model_im.train()
        log_print("\t Calculated lambda AE init: %f" % lambda_ae)

    for epoch in xrange(1, epochs+1):
        log_print("Epoch %i/%i" % (epoch, epochs))
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
                dataloader = val_dataloader
            if dataloader.sampler is None:
                num_images = len(dataloader.dataset)
            else:
                num_images = len(dataloader.sampler)
            log_print("\t[%s] - %i ims, %i batches" % (phase, num_images, num_images/config.BATCH_SIZE))


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
                loss_ae = lambda_ae * loss_ae_f(out_voxels, voxels)
                loss_im = torch.sum((im_embed - voxel_embed)**2) / im_embed.data.nelement()
                total_loss = loss_ae + loss_im
                curr_loss_ae += voxels.size(0) * loss_ae.item()
                curr_loss_im += voxels.size(0) * loss_im.item()

                # Backprop and cleanup
                if phase == "train":
                    total_loss.backward()
                    optimizer.step()
                else:
                    model_ae.train()
                    model_im.train()

                # Calculate accuracy (IOU)
                voxels = voxels.detach()
                out_voxels = out_voxels.detach()
                iou = calc_iou_acc(voxels, out_voxels, config.IOU_THRESH)
                curr_iou += voxels.size(0) * iou

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
            epoch_loss_ae = float(epoch_loss_ae + curr_loss_ae) / float(num_images)
            epoch_loss_im = float(epoch_loss_im + curr_loss_im) / float(num_images)
            total_epoch_loss = epoch_loss_ae + epoch_loss_im
            epoch_iou = float(epoch_iou + curr_iou) / float(num_images)
            log_print("\tEPOCH %i [%s] - Avg Loss: %f | Avg AE Loss: %f , Avg Im Loss: %f , Avg IoU: %f" % (epoch, phase, total_epoch_loss, epoch_loss_ae, epoch_loss_im, epoch_iou))

            # Save best model weights
            err_improvement = best_loss - total_epoch_loss
            if phase == "val":
                if err_improvement >= 0:
                    if (not model_ae.module.training) or (not model_im.module.training):
                        log_print("ERROR! Must be in training mode before saving weights")
                        model_ae.train()
                        model_im.train()
                    log_print("\tBEST NEW EPOCH: %i" % epoch)
                    best_loss = total_epoch_loss
                    best_model_ae.load_state_dict(model_ae.state_dict())
                    best_model_im.load_state_dict(model_im.state_dict())
                    best_epoch = epoch
                    save_model_weights(model_ae, "%s_ae3d" % config.JOINT_RUN_NAME)
                    save_model_weights(model_im, "%s_im" % config.JOINT_RUN_NAME)
                else:
                    log_print("\tCurrent best epoch: %i, %f loss" % (best_epoch, best_loss))

            # Checkpoint epoch weights
            if (phase == "val") and (epoch % epoch_checkpoint == 0):
                log_print("\tCheckpoint weights for epoch %i" % epoch)
                save_model_weights(model_ae, "%s_ae3d_%i" % (config.JOINT_RUN_NAME, epoch))
                save_model_weights(model_im, "%s_im_%i" % (config.JOINT_RUN_NAME, epoch))

    # Finish up
    time_elapsed = time.time() - init_time
    log_print("BEST EPOCH: %i/%i - Loss: %f" % (best_epoch, epochs, best_loss))
    log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
    del model_ae, model_im
    return (best_model_ae, best_model_im)

def model_view_step(model_ae, model_im, M_list, M_ind_map, dataloader, lambda_ae, lambda_view, train=False, optimizer=None):

    # Init metrics
    init_time = time.time()
    curr_sup_loss = curr_view_loss = 0.0
    total_sup_loss = total_view_loss = 0.0
    print_interval = config.VIEW_PRINT_INTERVAL
    batch_count = 1
    if dataloader.sampler is None:
        num_images = len(dataloader.dataset)
    else:
        num_images = len(dataloader.sampler)

    if train:
        model_ae.train()
        model_im.train()
    else:
        model_ae.val()
        model_im.val()
    loss_ae_f = nn.BCELoss().cuda()

    epoch_loss_sup_ae = 0.0
    epoch_loss_sup_im = 0.0
    epoch_loss_view = 0.0

    # Train the epoch
    log_print("\t\t[%s] - %i ims, %i batches" % ("TRAIN" if train else "VAL", num_images, num_images/config.BATCH_SIZE))
    for data in dataloader:

        # Gather data
        ims, voxels, im_names, domains = \
            data['im'], data['voxel'], data['im_name'], np.array(data['domain'])
        if config.GPU and torch.cuda.is_available():
            ims = ims.cuda()
            voxels = voxels.cuda()
        ims = Variable(ims).float()
        voxels = Variable(voxels).float()

        # Forward pass
        if train and (optimizer is not None):
            optimizer.zero_grad()
        im_embed = model_im(ims)
        voxel_embed = model_ae.module._encode(voxels)
        out_voxels_ae = model_ae.module._decode(voxel_embed)
        out_voxels_im = model_ae.module._decode(im_embed)

        # Compute masks for src and target data
        src_inds = np.where(domains == 'src')[0]
        target_inds = np.where(domains == 'target')[0]

        # Source data only
        im_embed_masked_src = im_embed.clone()
        voxel_embed_masked_src = voxel_embed.clone()
        out_voxels_ae_masked_src = out_voxels_ae.clone()
        voxels_masked_src = voxels.clone()
        im_embed_masked_src[target_inds] = 0
        voxel_embed_masked_src[target_inds] = 0
        out_voxels_ae_masked_src[target_inds] = 0
        voxels_masked_src[target_inds] = 0

        # Target data only
        im_embed_masked_target = im_embed.clone()
        voxel_embed_masked_target = voxel_embed.clone()
        out_voxels_im_masked_target = out_voxels_im.clone()
        voxels_masked_target = voxels.clone()
        im_embed_masked_target[src_inds] = 0
        voxel_embed_masked_target[src_inds] = 0
        out_voxels_im_masked_target[src_inds] = 0
        voxels_masked_target[src_inds] = 0

        # Compute supervised loss (src domain)
        loss_sup_ae = \
            (float(im_embed.size(0))*(loss_ae_f(out_voxels_ae_masked_src, voxels_masked_src)))/float(len(src_inds))
        loss_sup_im = \
            torch.sum((im_embed_masked_src-voxel_embed_masked_src)**2) / float(len(src_inds))
        loss_sup = (lambda_ae * loss_sup_ae) + loss_sup_im

        # Compute view consistency loss (target domain)
        M_masked_target = \
            Variable(torch.zeros(out_voxels_im_masked_target.size()).cuda(), requires_grad=False)
        for target_ind in target_inds:
            target_id = str(im_names[target_ind]).split("_")[1]
            M_masked_target[target_ind] = M_list[M_ind_map[target_id]]
        loss_view = \
            torch.sum((out_voxels_im_masked_target-M_masked_target)**2) / float(len(target_inds))

        # Backprop and cleanup
        curr_loss_sup_ae = (lambda_ae * loss_sup_ae).data.item()
        curr_loss_sup_im = loss_sup_im.data.item()
        curr_loss_view = (lambda_view * loss_view).data.item()
        total_loss = loss_sup + (lambda_view * loss_view)
        if train and (optimizer is not None):
            total_loss.backward()
            optimizer.step()

        # Output
        #TODO
        if batch_count % print_interval == 0:
            epoch_loss_sup_ae += curr_loss_sup_ae 
            epoch_loss_sup_im += curr_loss_sup_im 
            epoch_loss_view += curr_loss_view
            if train:
                one = float(curr_loss_sup_ae) / float(print_interval * config.BATCH_SIZE)
                two = float(curr_loss_sup_im) / float(print_interval * config.BATCH_SIZE)
                three = float(curr_loss_view) / float(print_interval * config.BATCH_SIZE)
                total = one + two + three
                log_print("\t\tBatches %i-%i -\tAvg Total: %f, Avg Sup AE: %f, Avg Sup Im: %f, Avg View: %f" % (batch_count-print_interval+1, batch_count, total, one, two, three))
            curr_loss_sup_ae = curr_loss_sup_im = curr_loss_view = 0
        batch_count += 1

    # Report epoch results
    one = float(epoch_loss_sup_ae + curr_loss_sup_ae) / float(num_images)
    two = float(epoch_loss_sup_im + curr_loss_sup_im) / float(num_images)
    three = float(epoch_loss_view + curr_loss_view) / float(num_images)
    total = one + two + three
    log_print("\t\tEPOCH RESULTS - Avg Total: %f, Avg Sup AE: %f, Avg Sup Im: %f, Avg View: %f" % (total, one, two, three))

    # Return
    model_ae.train()
    model_im.train()
    return model_ae, model_im 

#######################
#   TRAINING MAINS

def train_autoencoder():
    # Create training DataLoader
    log_print("Loading training data...")
    train_dataloader =  \
        create_shapenet_voxel_dataloader(
            "train",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE,
            subset_size_=config.AE_SUBSET_SIZE_TRAIN)
    val_dataloader = \
        create_shapenet_voxel_dataloader(
            "val",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE,
            subset_size_=config.AE_SUBSET_SIZE_VAL)

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
            config.BATCH_SIZE,
            subset_size_=config.IM_SUBSET_SIZE_TRAIN)
    val_dataloader = \
        create_image_network_dataloader(
            "val",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE,
            subset_size_=config.IM_SUBSET_SIZE_VAL)

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
            config.BATCH_SIZE,
            subset_size_=config.JOINT_SUBSET_SIZE_TRAIN)
    val_dataloader = \
        create_image_network_dataloader(
            "val",
            config.DATA_BASE_DIR,
            config.BATCH_SIZE,
            subset_size_=config.JOINT_SUBSET_SIZE_VAL)

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

def train_view():

    # Create datasets
    train_dataloader = \
        create_view_consistency_dataloader(
            "train",
            config.VIEW_SRC_DATA_BASE_DIR,
            config.VIEW_TARGET_DATA_BASE_DIR,
            config.BATCH_SIZE,
            src_sample_mult=config.VIEW_SRC_SAMPLE_MULTIPLIER)

    # Initialize network weights
    log_print("Creating image network and 3d-autoencoder models...")
    model_ae = create_model_3d_autoencoder(config.VOXEL_RES, config.EMBED_SIZE)
    model_im = create_model_image_network(config.EMBED_SIZE)
    load_success_ae = try_load_weights(model_ae, config.VIEW_AE3D_LOAD_WEIGHTS)
    load_success_im = try_load_weights(model_im, config.VIEW_IM_LOAD_WEIGHTS)
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
        if not (try_load_weights(model_ae, config.VIEW_AE3D_LOAD_WEIGHTS)):
            log_print("FAILED TO LOAD AE3D WEIGHTS")
    if not load_success_im:
        if not try_load_weights(model_im, config.VIEW_IM_LOAD_WEIGHTS):
            log_print("FAILED TO LOAD IMAGE NETWORK WEIGHTS")

    # Set up loss and optimizer
    loss_ae = nn.BCELoss()
    if config.GPU and torch.cuda.is_available():
        loss_ae = loss_ae.cuda()
    params = list(model_ae.parameters()) + \
        list(filter(lambda p: p.requires_grad, model_im.parameters()))
    optimizer = optim.SGD(
        params,
        lr=config.VIEW_LEARNING_RATE,
        momentum=config.VIEW_MOMENTUM)
    explorer = None
    lambda_ae = 1.0 #TODO
    lambda_view = 1.0 #TODO

    # Fetch Ys
    log_print("Fetching source domain ground truths...")
    train_src_dataloader = \
        create_shapenet_voxel_dataloader(
            'train', 
            config.VIEW_SRC_DATA_BASE_DIR, 
            config.BATCH_SIZE)
    Y_list, Y_ind_map = optim_latent.fetch_Y(train_src_dataloader)
    tmp = ImageVoxelDataset('train', config.VIEW_SRC_DATA_BASE_DIR)
    Y_im_counts = tmp.im_counts.copy()
    del tmp
    log_print("\t%i source domain ground truth models" % len(Y_list))
    
    # Init latents
    log_print("Initializing latent variables...")
    train_target_dataloader = \
        create_dataloader_from_dataset(train_dataloader.dataset.target_dataset, 32)
    if config.VIEW_INIT_AVG is None:
        log_print("\tCalculating d(*,*) avg")
        avg = optim_latent.calc_avg(model_ae, model_im, train_target_dataloader, Y_list)
    else:
        avg = config.VIEW_INIT_AVG
    log_print("\tMean d(*,*): %f" % avg)
    M_list, M_ind_map = optim_latent.init_latents(
        model_ae, model_im, train_target_dataloader, Y_list, Y_ind_map, Y_im_counts, avg*avg)
    del train_src_dataloader, train_target_dataloader

    log_print("Starting optimization!!!")
    for iter_ in xrange(1, config.VIEW_EPOCHS+1):
        log_print("\tEpoch: %i/%i" % (iter_, config.VIEW_EPOCHS+1))

        # Recreate dataloader (new subset of src data)
        log_print("\t  Refreshing src training data")
        train_dataloader = \
            create_fusion_dataloader(
                train_dataloader.dataset, config.BATCH_SIZE, config.VIEW_SRC_SAMPLE_MULTIPLIER)

        # Fix latent, optimize network
        for e in xrange(config.VIEW_INNER_EPOCHS):
            log_print("\t  Optimizing network parameters G():")
            model_ae, model_im = model_view_step(
                model_ae, model_im, M_list, M_ind_map, train_dataloader, lambda_ae, lambda_view, train=True, optimizer=optimizer)

        # Fix network, optimize latent
        log_print("\t  Optimizing latent configs M:")
        M_list = optim_latent.update_latents(M_list) #TODO

        # Checkpoint weights
        log_print("\t  Saving epoch %i weights" % iter_)
        save_model_weights(model_ae, "%s_ae3d_%i" % (config.VIEW_RUN_NAME, iter_))
        save_model_weights(model_im, "%s_im_%i" % (config.VIEW_RUN_NAME, iter_))

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

    # Use cudnn bencharmking
    cudnn.benchmark = True

    # Run 3 step training process
    log_print("***BEGINNING PART 1: train 3D autoencoder")
    train_autoencoder()
    log_print("***FINISHED PART 1") 

    log_print("***BEGINNING PART 2: train image network")
    train_image_network()
    log_print("***FINISHED PART 2") 

    log_print("***BEGINNING PART 3: joint training")
    train_joint()
    log_print("***FINISHED PART 3") 

    if config.VIEW_INCLUDE:
        log_print("***BEGINNING PART 4: view consistency adaptation")
        train_view()
        log_print("***FINISHED PART 4")

    # Finished
    log_print("Script DONE!")

if __name__ == "__main__":
    main()
