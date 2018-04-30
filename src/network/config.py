
LOG_FILE = True

# Filepaths
RUN_NAME = "debug-view"
RUN_DESCRIPTION = "debug-view"
PROJ_BASE_DIR = "../.."
OBJECT_CLASS = "CHAIR"
DATA_BASE_DIR = "%s/data/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_BASE_DIR = "%s/output/%s" % (PROJ_BASE_DIR, OBJECT_CLASS)
OUT_WEIGHTS_DIR = "%s/models/%s" % (OUT_BASE_DIR, RUN_NAME)
OUT_LOG_FP = "%s/logs/%s.log" % (OUT_BASE_DIR, RUN_NAME)

# Program parameters
GPU = True
MULTI_GPU = True
BATCH_SIZE = 32
VOXEL_RES = 20
IOU_THRESH = 0.5
EMBED_SIZE = 64

#####################################################################
#   3D AUTOENCODER PARAMS
AE_RUN_NAME = "ae3d"
AE_INIT_WEIGHTS = "../../output/CHAIR/models/tl-default-final/joint_ae3d.pt"
AE_LEARNING_RATE = 1e-3
AE_EPOCHS = 0
AE_WEIGHTS_CHECKPOINT = 50
AE_PRINT_INTERVAL = 20
AE_SUBSET_SIZE_TRAIN = None
AE_SUBSET_SIZE_VAL= None

#####################################################################
#   IMAGE NETWORK PARAMS
IM_RUN_NAME = "im-network"
IM_INIT_WEIGHTS = "../../output/CHAIR/models/tl-default-final/joint_im.pt"
IM_AE3D_LOAD_WEIGHTS = "%s/models/%s/%s.pt" % (OUT_BASE_DIR, RUN_NAME, AE_RUN_NAME)
IM_LEARNING_RATE = 1e-4
IM_MOMENTUM = 0.9
IM_EPOCHS = 0
IM_WEIGHTS_CHECKPOINT = 20
IM_PRINT_INTERVAL = 50
IM_SUBSET_SIZE_TRAIN = None
IM_SUBSET_SIZE_VAL= None

####################################################################
#   JOINT TRAINING PARAMS
JOINT_RUN_NAME = "joint"
JOINT_AE3D_LOAD_WEIGHTS = "%s/models/%s/%s.pt" % (OUT_BASE_DIR, RUN_NAME, AE_RUN_NAME)
JOINT_IM_LOAD_WEIGHTS = "%s/models/%s/%s.pt" % (OUT_BASE_DIR, RUN_NAME, IM_RUN_NAME)
JOINT_LEARNING_RATE = 1e-5
JOINT_MOMENTUM = 0.9
JOINT_EPOCHS = 0
JOINT_WEIGHTS_CHECKPOINT = 20
JOINT_PRINT_INTERVAL = 50
JOINT_SUBSET_SIZE_TRAIN = None
JOINT_SUBSET_SIZE_VAL= None

####################################################################
#   VIEW CONSISTENCY PARAMS
VIEW_INCLUDE = True
VIEW_RUN_NAME = "view"
VIEW_SRC_DATA_BASE_DIR = DATA_BASE_DIR
VIEW_TARGET_DATA_BASE_DIR = "%s/data/RedwoodRGB_Chair" % PROJ_BASE_DIR
VIEW_AE3D_LOAD_WEIGHTS = "%s/models/%s/%s_ae3d.pt" % (OUT_BASE_DIR, RUN_NAME, JOINT_RUN_NAME)
VIEW_IM_LOAD_WEIGHTS = "%s/models/%s/%s_im.pt" % (OUT_BASE_DIR, RUN_NAME, JOINT_RUN_NAME)
VIEW_LEARNING_RATE = 1e-5
VIEW_MOMENTUM = 0.9
VIEW_EPOCHS = 3
VIEW_INNER_EPOCHS = 1
VIEW_SRC_SAMPLE_MULTIPLIER = 1.0
VIEW_PRINT_INTERVAL = 50
VIEW_INIT_AVG = 0.295183
VIEW_LAMBDA_VIEW = 20.0
VIEW_LAMBDA_ALIGN = 0.0
VIEW_VOXELIZE_THRESH = 0.2
VIEW_SUBSET_SIZE_TRAIN = None
VIEW_SUBSET_SIZE_VAL = None
VIEW_INIT_LATENTS = "./init_latents.mat"

#####################################################################
# Debugging print method
def PRINT_CONFIG():
    print "~~~~~ BEGIN CONFIG ~~~~~"
    print " "

    print "~~Filepaths~~"
    print "RUN_NAME:\t\t\t", RUN_NAME
    print "RUN_DESCRIPTION:\t\t", RUN_DESCRIPTION
    print "PROJ_BASE_DIR:\t\t\t", PROJ_BASE_DIR 
    print "OBJECT_CLASS:\t\t\t", OBJECT_CLASS 
    print "DATA_BASE_DIR:\t\t\t", DATA_BASE_DIR 
    print "OUT_BASE_DIR:\t\t\t", OUT_BASE_DIR 
    print "OUT_WEIGHTS_DIR:\t\t", OUT_WEIGHTS_DIR 
    print "OUT_LOG_FP:\t\t\t", OUT_LOG_FP 
    print " "

    print "~~Program parameters~~"
    print "GPU:\t\t\t\t", GPU 
    print "MULTI_GPU:\t\t\t", MULTI_GPU 
    print "BATCH_SIZE:\t\t\t", BATCH_SIZE 
    print "VOXEL_RES:\t\t\t", VOXEL_RES 
    print "IOU_THRESH:\t\t\t", IOU_THRESH 
    print "EMBED_SIZE:\t\t\t", EMBED_SIZE 
    print " "

    print "~~3D Autoencoder Params~~"
    print "AE_RUN_NAME:\t\t\t", AE_RUN_NAME 
    print "AE_INIT_WEIGHTS:\t\t", AE_INIT_WEIGHTS 
    print "AE_LEARNING_RATE:\t\t", AE_LEARNING_RATE 
    print "AE_EPOCHS:\t\t\t", AE_EPOCHS 
    print "AE_WEIGHTS_CHECKPOINT:\t\t", AE_WEIGHTS_CHECKPOINT 
    print "AE_PRINT_INTERVAL:\t\t", AE_PRINT_INTERVAL
    print "AE_SUBSET_SIZE_TRAIN:\t\t", AE_SUBSET_SIZE_TRAIN
    print "AE_SUBSET_SIZE_VAL:\t\t", AE_SUBSET_SIZE_VAL

    print " "

    print "~~Image Network Params~~"
    print "IM_RUN_NAME:\t\t\t", IM_RUN_NAME 
    print "IM_AE3D_LOAD_WEIGHTS:\t\t", IM_AE3D_LOAD_WEIGHTS 
    print "IM_INIT_WEIGHTS:\t\t", IM_INIT_WEIGHTS 
    print "IM_LEARNING_RATE:\t\t", IM_LEARNING_RATE 
    print "IM_MOMENTUM:\t\t\t", IM_MOMENTUM
    print "IM_EPOCHS:\t\t\t", IM_EPOCHS 
    print "IM_WEIGHTS_CHECKPOINT:\t\t", IM_WEIGHTS_CHECKPOINT 
    print "IM_PRINT_INTERVAL:\t\t", IM_PRINT_INTERVAL
    print "IM_SUBSET_SIZE_TRAIN:\t\t", IM_SUBSET_SIZE_TRAIN
    print "IM_SUBSET_SIZE_VAL:\t\t", IM_SUBSET_SIZE_VAL

    print " "

    print "~~Joint Training Params~~"
    print "JOINT_RUN_NAME:\t\t\t", JOINT_RUN_NAME 
    print "JOINT_AE3D_LOAD_WEIGHTS:\t", JOINT_AE3D_LOAD_WEIGHTS 
    print "JOINT_IM_LOAD_WEIGHTS:\t\t", JOINT_IM_LOAD_WEIGHTS 
    print "JOINT_LEARNING_RATE:\t\t", JOINT_LEARNING_RATE 
    print "JOINT_MOMENTUM:\t\t\t", JOINT_MOMENTUM
    print "JOINT_EPOCHS:\t\t\t", JOINT_EPOCHS 
    print "JOINT_WEIGHTS_CHECKPOINT:\t", JOINT_WEIGHTS_CHECKPOINT 
    print "JOINT_PRINT_INTERVAL:\t\t", JOINT_PRINT_INTERVAL
    print "JOINT_SUBSET_SIZE_TRAIN:\t\t", IM_SUBSET_SIZE_TRAIN
    print "JOINT_SUBSET_SIZE_VAL:\t\t", IM_SUBSET_SIZE_VAL
    print ""

    print "~~View consistency params~~"
    print "VIEW_INCLUDE:\t", VIEW_INCLUDE
    print "VIEW_RUN_NAME:\t", VIEW_RUN_NAME
    print "VIEW_SRC_DATA_BASE_DIR:\t", VIEW_SRC_DATA_BASE_DIR
    print "VIEW_TARGET_DATA_BASE_DIR:\t", VIEW_TARGET_DATA_BASE_DIR
    print "VIEW_AE3D_LOAD_WEIGHTS:\t", VIEW_AE3D_LOAD_WEIGHTS
    print "VIEW_IM_LOAD_WEIGHTS:\t", VIEW_IM_LOAD_WEIGHTS
    print "VIEW_LEARNING_RATE:\t", VIEW_LEARNING_RATE
    print "VIEW_MOMENTUM:\t", VIEW_MOMENTUM
    print "VIEW_EPOCHS:\t", VIEW_EPOCHS
    print "VIEW_INNER_EPOCHS:\t", VIEW_INNER_EPOCHS
    print "VIEW_SRC_SAMPLE_MULTIPLIER:\t", VIEW_SRC_SAMPLE_MULTIPLIER
    print "VIEW_PRINT_INTERVAL:\t", VIEW_PRINT_INTERVAL
    print "VIEW_INIT_AVG:\t", VIEW_INIT_AVG
    print "VIEW_LAMBDA_VIEW:\t", VIEW_LAMBDA_VIEW
    print "VIEW_LAMBDA_ALIGN:\t", VIEW_LAMBDA_ALIGN
    print "VIEW_SUBSET_SIZE_TRAIN:\t", VIEW_SUBSET_SIZE_TRAIN
    print "VIEW_SUBSET_SIZE_VAL:\t", VIEW_SUBSET_SIZE_VAL
    print "VIEW_INIT_LATENTS:\t", VIEW_INIT_LATENTS
    print " "

    print "~~~~~~ END CONFIG ~~~~~~"
