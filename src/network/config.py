# Filepaths
RUN_NAME = "test"
RUN_DESCRIPTION = "test"
PROJ_BASE_DIR = "../.."
OBJECT_CLASS = "CHAIR"
DATA_BASE_DIR = "%s/data/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_BASE_DIR = "%s/output/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_WEIGHTS_FP = "%s/models/%s.pt" % (OUT_BASE_DIR, RUN_NAME)
OUT_LOG_FP = "%s/logs/%s.log" % (OUT_BASE_DIR, RUN_NAME)
OUT_PRED_FP = "%s/preds/%s.pred" % (OUT_BASE_DIR, RUN_NAME)

# Program parameters
GPU = True
MULTI_GPU = True
VOXEL_RES = 20

# Learning parameters
NETWORK_TYPE = "SIMPLE"
RESNET_LAYERS = 18 #18, 34, 50, 101, 152
PRETRAINED = True
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.01
MOMENTUM = 0.9
STEP_SIZE = 4
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
  print "OUT_WEIGHTS_FP:\t", OUT_WEIGHTS_FP
  print "OUT_LOG_FP:\t", OUT_LOG_FP
  print "OUT_PRED_FP:\t", OUT_PRED_FP
  print " "

  print "~~PROGRAM PARAMS~~"
  print "GPU:\t\t", GPU
  print "MULTI_GPU:\t", MULTI_GPU
  print "VOXEL_RES:\t", VOXEL_RES
  print ""

  print "~~LEARNING PARAMS~~"
  print "NETWORK_TYPE:\t", NETWORK_TYPE
  print "RESNET_LAYERS:\t", RESNET_LAYERS
  print "PRETRAINED:\t", PRETRAINED
  print "BATCH_SIZE:\t", BATCH_SIZE
  print "EPOCHS:\t\t", EPOCHS
  print "LEARNING_RATE:\t", LEARNING_RATE
  print "MOMENTUM:\t", MOMENTUM
  print "STEP_SIZE:\t", STEP_SIZE
  print "GAMMA:\t\t", GAMMA
  print "EMBED_SIZE:\t", EMBED_SIZE
  print ""

  print "~~~~~~ END CONFIG ~~~~~~"
