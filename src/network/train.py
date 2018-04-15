import os
import sys
import math
import config
import datetime
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

# Imports from src files
from datasets import ShapeNetVoxelDataset

#####################
#   BEGIN HELPERS   #
#####################

def log_print(string):
    print "[%s]\t %s" % (datetime.datetime.now(), string)

def create_shapenet_voxel_dataloader(dset_type_, data_base_dir_, batch_size_):
    dataset = ShapeNetVoxelDataset( \
        dset_type=dset_type_, \
        data_base_dir=data_base_dir_, \
        transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = DataLoader( \
        dataset, \
        batch_size=batch_size_, \
        shuffle=True, \
        num_workers=4)
    return dataloader

def create_model(network_type):
  res_v = config.RESNET_LAYERS

  if network_type == "VIEWPOINT":
    model = models.viewpoint_net(layers=res_v, pretrained=config.PRETRAINED)
  elif network_type == "VIEWPOINT_CLASS_DOMAIN":
    model = models.vcd_net(layers=res_v, pretrained=config.PRETRAINED)
    model.fc2_class = nn.Linear(model.fc2_class.in_features, config.NUM_OBJ_CLASSES)
  elif network_type == "SIMPLE":
    model = models.resnet_experiment(bottleneck_size=config.BOTTLENECK_SIZE, layers=res_v, pretrained=config.PRETRAINED)

  # Adjust network size
  model.fc_azi = nn.Linear(model.fc_azi.in_features, config.AZIMUTH_BINS)
  model.fc_ele = nn.Linear(model.fc_ele.in_features, config.ELEVATION_BINS)
  return model

def train_model(model, train_dataloader, regular_dataloader, val_dataloader, loss_f_viewpoint, loss_f_mmd, optimizer, explorer, epochs):

  init_time = time.time()
  best_loss = best_az_err = best_ele_err = 0.0
  best_weights = model.state_dict().copy()
  best_epoch = -1

  for epoch in xrange(epochs):
    log_print("Epoch %i/%i: %i batches of %i images each" % (epoch+1, epochs, len(train_dataloader.dataset)/config.BATCH_SIZE, config.BATCH_SIZE))

    for phase in ["train", "val"]:
      if phase == "train":
        explorer.step()
        model.train(True)
        dataloader = train_dataloader
      else:
        model.train(False)
        dataloader = val_dataloader

      # Iterate over dataset
      epoch_loss = epoch_az_err = epoch_ele_err = 0
      curr_loss = curr_az_err = curr_ele_err = 0
      print_interval = 100
      batch_count = 0

      # Create regularizer dataloader iterator
      if regular_dataloader is not None and phase == "train":
        regular_iter = iter(regular_dataloader)

      for data in dataloader:

        # Gather batch data (images + corersponding annots)
        im_fps, inputs, annot_azimuths, annot_elevations, annot_classes, annot_domains= \
          data['image_fp'], data['image'], data['azimuth'], data['elevation'], data['class_id'], data['domain_id']

        # Wrap as pytorch autograd Variable
        if config.GPU and torch.cuda.is_available():
          inputs = Variable(inputs.cuda())
          annot_azimuths = Variable(annot_azimuths.cuda())
          annot_elevations = Variable(annot_elevations.cuda())
          annot_classes = Variable(annot_classes.cuda())
          annot_domains = Variable(annot_domains.cuda())
        else:
          inputs = Variable(inputs)
          annot_azimuths = Variable(annot_azimuths)
          annot_elevations = Variable(annot_elevations)
          annot_classes = Variable(annot_classes)
          annot_domains = Variable(annot_domains)

        # Forward pass and calculate loss
        optimizer.zero_grad()
        out_azimuths, out_elevations = model(inputs)

        # Regularizer
        if regular_dataloader is not None and phase == "train":
          embeddings = model.get_embedding()
          while True:
            regular_data = next(regular_iter)
            if regular_data['azimuth'].size(0) == config.BATCH_SIZE:
              break
            regular_iter = iter(regular_dataloader)

          regular_inputs = regular_data['image']
          if config.GPU and torch.cuda.is_available():
            regular_inputs = Variable(regular_inputs.cuda())
          else:
            regular_inputs = Variable(regular_inputs)
          _,_ = model(regular_inputs)
          regular_embeddings = model.get_embedding()

        # Calculate losses
        loss_azimuth = loss_f_viewpoint(out_azimuths, annot_azimuths)
        loss_elevation = loss_f_viewpoint(out_elevations, annot_elevations)
        if regular_dataloader is not None and phase == "train":
          loss_mmd = loss_f_mmd(embeddings, regular_embeddings)
        else:
          loss_mmd = 0
        loss = (loss_azimuth + loss_elevation) + (config.LAMBDA_MMD * loss_mmd)
        curr_loss += loss.data[0]

        # Update accuracy
        _, pred_azimuths = torch.max(out_azimuths.data, 1)
        azimuth_diffs = torch.abs(pred_azimuths - annot_azimuths.data)
        azimuth_errs = torch.min(azimuth_diffs, 360-azimuth_diffs)
        curr_az_err += azimuth_errs.sum()
        _, pred_elevations = torch.max(out_elevations.data, 1)
        elevation_diffs = torch.abs(pred_elevations - annot_elevations.data)
        elevation_errs = torch.min(elevation_diffs, 360-elevation_diffs)
        curr_ele_err += elevation_errs.sum()
        
        # Backward pass (if train)
        if phase == "train":
          loss.backward()
          optimizer.step()
          if regular_dataloader is not None:
            del embeddings, regular_embeddings

        # Output
        if batch_count % print_interval == 0 and batch_count != 0:
          epoch_loss += curr_loss
          epoch_az_err += curr_az_err
          epoch_ele_err += curr_ele_err
          if phase == "train":
            curr_loss = float(curr_loss) / float(print_interval*config.BATCH_SIZE)
            curr_az_err = float(curr_az_err) / float(print_interval*config.BATCH_SIZE)
            curr_ele_err = float(curr_ele_err) / float(print_interval*config.BATCH_SIZE)
            log_print("\tBatches %i-%i -\tLoss: %f \t Azimuth Err: %f   Elevation Err: %f" % (batch_count-print_interval+1, batch_count, curr_loss, curr_az_err, curr_ele_err))
          curr_loss = curr_az_err = curr_ele_err = 0
        batch_count += 1
      
      # Report epoch results
      num_images = len(dataloader.dataset)
      epoch_loss = float(epoch_loss+curr_loss) / float(num_images)
      epoch_az_err = float(epoch_az_err+curr_az_err) / float(num_images)
      epoch_ele_err = float(epoch_ele_err+curr_ele_err) / float(num_images)
      log_print("\tEPOCH %i [%s] - Loss: %f   Azimuth Err: %f   Elevation Err: %f" % (epoch+1, phase, epoch_loss, epoch_az_err, epoch_ele_err))

      # Save best model weights from epoch
      err_improvement = (best_az_err - epoch_az_err) + (best_ele_err - epoch_ele_err)
      if phase == "val" and (err_improvement >= 0 or epoch == 0):
        log_print("BEST NEW EPOCH: %i" % epoch)
        best_az_err = epoch_az_err
        best_ele_err = epoch_ele_err
        best_loss = epoch_loss
        best_weights = model.state_dict().copy()
        best_epoch = epoch

  # Finish up
  time_elapsed = time.time() - init_time
  log_print("BEST EPOCH: %i/%i - Loss: %f   Azimuth Err: %f   Elevation Err: %f" % (best_epoch+1, epochs, best_loss, best_az_err, best_ele_err))
  log_print("Training completed in %sm %ss" % (time_elapsed // 60, time_elapsed % 60))
  model.load_state_dict(best_weights)
  return model

def save_model_weights(model, filepath):
  torch.save(model.state_dict(), filepath)

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

    """
    # Set up model for training
    log_print("Creating model...")
    model = create_model(config.NETWORK_TYPE)
    if config.GPU and torch.cuda.is_available():
        log_print("Enabling GPU")
        if config.MULTI_GPU and torch.cuda.device_count() > 1:
            log_print("Using multiple GPUs: %i" % torch.cuda.device_count())
            model = nn.DataParallel(model)
        model = model.cuda()
    else:
        log_print("Ignoring GPU (CPU only)")

    # Set up loss and optimizer
    loss_f_viewpoint = nn.CrossEntropyLoss()
    loss_f_mmd = MMDLoss()
    if config.GPU and torch.cuda.is_available():
        loss_f_viewpoint  = loss_f_viewpoint.cuda()
        loss_f_mmd = loss_f_mmd.cuda()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        momentum=config.MOMENTUM)
    explorer = lr_scheduler.StepLR(
        optimizer, 
        step_size=config.STEP_SIZE,
        gamma=config.GAMMA)

    # Perform training
    log_print("!!!!!Starting training!!!!!")
    model = train_model(model, train_dataloader, regular_dataloader, val_dataloader, loss_f_viewpoint, loss_f_mmd, optimizer, explorer, config.EPOCHS)
  
    # Save model weights
    log_print("Saving model weights to %s..." % config.OUT_WEIGHTS_FP)
    save_model_weights(model, config.OUT_WEIGHTS_FP)
    """

    log_print("Script DONE!")

if __name__ == "__main__":
    main()
