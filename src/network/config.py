# Filepaths
RUN_NAME = "chair-ae3d-long2"
RUN_DESCRIPTION = "training 3d auto-encoder on chair voxels. long version (200 epochs w/ low learning rate)."
PROJ_BASE_DIR = "../.."
OBJECT_CLASS = "CHAIR"
DATA_BASE_DIR = "%s/data/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_BASE_DIR = "%s/output/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_WEIGHTS_DIR = "%s/models/%s" % (OUT_BASE_DIR, RUN_NAME)
OUT_LOG_FP = "%s/logs/%s.log" % (OUT_BASE_DIR, RUN_NAME)

# Program parameters
GPU = True
MULTI_GPU = True
WEIGHTS_CHECKPOINT = 20
VOXEL_RES = 20
IOU_THRESH = 0.5

# Learning parameters
BATCH_SIZE = 32
EPOCHS = 250
LEARNING_RATE = 1e-2
LR_STEPS = [10, 20, 40, 120]
MOMENTUM = 0.9
STEP_SIZE = 20
GAMMA = 0.1
EMBED_SIZE = 64

# Debugging print method
def PRINT_CONFIG():
  print "~~~~~ BEGIN CONFIG ~~~~~"
  print " "

  print "~~FILEPATHS~~"
  print "RUN_NAME:\t", RUN_NAME
  print "RUN_DESCRIP:\t", RUN_DESCRIPTION
  print "PROJ_BASE_DIR:\t", PROJ_BASE_DIR
  print "OBJECT_CLASS:\t", OBJECT_CLASS
  print "DATA_BASE_DIR:\t", DATA_BASE_DIR
  print "OUT_BASE_DIR:\t", OUT_BASE_DIR
  print "OUT_WEIGHTS_DIR:", OUT_WEIGHTS_DIR
  print "OUT_LOG_FP:\t", OUT_LOG_FP
  print " "

  print "~~PROGRAM PARAMS~~"
  print "GPU:\t\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "WEIGHTS_CHECKPOINT:", WEIGHTS_CHECKPOINT
  print "VOXEL_RES:\t", VOXEL_RES
  print "IOU_THRESH:\t", IOU_THRESH
  print ""

  print "~~LEARNING PARAMS~~"
  print "BATCH_SIZE:\t", BATCH_SIZE
  print "EPOCHS:\t\t", EPOCHS
  print "LEARNING_RATE:\t", LEARNING_RATE
  print "MOMENTUM:\t", MOMENTUM
  print "STEP_SIZE:\t", STEP_SIZE
  print "GAMMA:\t\t", GAMMA
  print "EMBED_SIZE:\t", EMBED_SIZE
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"
