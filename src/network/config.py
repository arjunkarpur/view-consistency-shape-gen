
# Filepaths
RUN_NAME = "imagenetwork-test"
RUN_DESCRIPTION = "test imagenetwork"
PROJ_BASE_DIR = "../.."
OBJECT_CLASS = "CHAIR"
DATA_BASE_DIR = "%s/data/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_BASE_DIR = "%s/output/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_WEIGHTS_DIR = "%s/models/%s" % (OUT_BASE_DIR, RUN_NAME)
OUT_LOG_FP = "%s/logs/%s.log" % (OUT_BASE_DIR, RUN_NAME)

#####################################################################
#   3D AUTOENCODER PARAMS

# Program parameters
GPU = True
MULTI_GPU = True
WEIGHTS_CHECKPOINT = 20
VOXEL_RES = 20
IOU_THRESH = 0.5

# Learning parameters
LOAD_WEIGHTS = None
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1.0
LR_STEPS = None
STEP_SIZE = None
MOMENTUM = 0.9
GAMMA = 0.1
EMBED_SIZE = 64

#####################################################################
#   IMAGE NETWORK PARAMS
IM_LEARNING_RATE = 1e-8
IM_MOMENTUM=0.9
IM_RUN_NAME = "im-network-1"
AE3D_LOAD_WEIGHTS = "%s/models/chair-ae3d-long6/chair-ae3d-long6.pt" % OUT_BASE_DIR

#####################################################################
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
  print "LOAD_WEIGHTS:\t", LOAD_WEIGHTS
  print "BATCH_SIZE:\t", BATCH_SIZE
  print "EPOCHS:\t\t", EPOCHS
  print "LEARNING_RATE:\t", LEARNING_RATE
  print "LR_STEPS:\t", LR_STEPS
  print "MOMENTUM:\t", MOMENTUM
  print "STEP_SIZE:\t", STEP_SIZE
  print "GAMMA:\t\t", GAMMA
  print "EMBED_SIZE:\t", EMBED_SIZE
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"
