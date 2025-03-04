from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Using cuda or cpu for training
_C.MODEL.DEVICE = "cuda"
# ID number of GPU
_C.MODEL.DEVICE_ID = '0'
# Name of backbone
_C.MODEL.NAME = 'resnet50'
# Last stride of backbone
_C.MODEL.LAST_STRIDE = 1
# Path to pretrained model of backbone
_C.MODEL.PRETRAIN_PATH = ''
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' , 'self' , 'finetune'
_C.MODEL.PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.NECK = 'bnneck'
# If train with multi-gpu ddp mode, options: 'True', 'False'
_C.MODEL.DIST_TRAIN = False
# If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.NO_MARGIN = False

# If train with arcface loss, options: 'True', 'False'
_C.MODEL.COS_LAYER = False
# depth of the model (added for resnet dynamic creation)
_C.MODEL.DEPTH = 50


# Transformer setting
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DROP_OUT = 0.0
_C.MODEL.ATT_DROP_RATE = 0.0
_C.MODEL.TRANSFORMER_TYPE = 'None'
_C.MODEL.STRIDE_SIZE = [16, 16]

# JPM Parameter
# _C.MODEL.JPM = False
_C.MODEL.SHIFT_NUM = 5
_C.MODEL.SHUFFLE_GROUP = 2
_C.MODEL.DEVIDE_LENGTH = 5
_C.MODEL.RE_ARRANGE = True

# Side Information Embedding Parameter
_C.MODEL.SIE_COEFFICIENT = 3.0
_C.MODEL.SIE_CAMERA = False
_C.MODEL.SIE_VIEW = False

# Sample Strategy
_C.MODEL.TRAIN_STRATEGY = '' # ['multiview', 'chunk']
_C.MODEL.SPATIAL = False
_C.MODEL.TEMPORAL = False
_C.MODEL.FREEZE = False
_C.MODEL.PYRAMID0_TYPE = ''
_C.MODEL.PYRAMID1_TYPE = ''
_C.MODEL.PYRAMID2_TYPE = ''
_C.MODEL.PYRAMID3_TYPE = ''
_C.MODEL.PYRAMID4_TYPE = ''
_C.MODEL.LAYER_COMBIN = 1
_C.MODEL.LAYER0_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER1_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER2_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER3_DIVISION_TYPE = 'NULL'
_C.MODEL.LAYER4_DIVISION_TYPE = 'NULL'
_C.MODEL.DIVERSITY = False



# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [384, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [384, 128]
# Random probability for image horizontal flip
_C.INPUT.HF_PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Random erasing
_C.INPUT.RE = True
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10
# indexes of data provided by train dataloader
_C.INPUT.TRAIN_KEYS = [0, 1, 2, 3]
# indexes of data provided by validation dataloader
_C.INPUT.EVAL_KEYS = [0, 1, 2, 3]
# index of person id in dataset
_C.INPUT.PERSON_ID_KEY = 1
# index of camera id in dataset
_C.INPUT.CAMERA_ID_KEY = 2


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.NAMES = ('market1501')
# Root directory where datasets should be used (and downloaded if not found)
# _C.DATASETS.ROOT_DIR = '/home/Datasets'
_C.DATASETS.ROOT_DIR = 'D:\datasets'
# folder where images are stored
_C.DATASETS.DIR = ('market1501')


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# Sampler for data loading
_C.DATALOADER.SAMPLER = 'softmax'
# Number of instance for one batch
_C.DATALOADER.NUM_INSTANCE = 16
# random select p persons for each sample
_C.DATALOADER.P = 16
# random select k tracklets for each person
_C.DATALOADER.K = 8
# random select 8 images of each tracklet for test
_C.DATALOADER.NUM_TEST_IMAGES = 8
# random select 8 images of each tracklet for train
_C.DATALOADER.NUM_TRAIN_IMAGES = 8

_C.DATALOADER.TRAIN_TRANSFORMS = [
    {'tranform':'resize'}, 
    {'tranform':'random_horizontal_flip', 'prob': 0.5}, 
    {'tranform':'pad', 'padding': 10}, 
    {'tranform':'random_crop'}, 
    {'tranform':'to_tensor'}, 
    {'tranform':'normalize'}, 
    {'tranform':'random_erasing', 'prob': 0.5}
    ]

_C.DATALOADER.TEST_TRANSFORMS = [
    {'tranform':'resize'}, 
    {'tranform':'to_tensor'}, 
    {'tranform':'normalize'}, 
    ]

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "Adam"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 100
# Base learning rate
_C.SOLVER.BASE_LR = 3e-4
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Factor of learning bias
_C.SOLVER.SEED = 1234
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.MARGIN = None
# Learning rate of SGD to learn the centers of center loss
_C.SOLVER.CENTER_LR = 0.5
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0005

# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = None
# scheduler type
_C.SOLVER.SCHEDULER = 'cosine'
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"
# iterations of warm up
_C.SOLVER.WARMUP_ITERS = 0

_C.SOLVER.COSINE_MARGIN = 0.5
_C.SOLVER.COSINE_SCALE = 30

# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
# contain 16 images per batch
_C.SOLVER.IMS_PER_BATCH = 64
# feature_dim
_C.SOLVER.FEATURE_DIMENSION = 2048

# ---------------------------------------------------------------------------- #
# Loss
# ---------------------------------------------------------------------------- #
_C.LOSS = CN()

# metric loss
_C.LOSS.METRIC_LOSS_TYPE = 'triplet'
# output tensor index
_C.LOSS.METRIC_LOSS_OUTPUT_INDEX = 1
# Id loss type, options: 'softmax','triplet'
_C.LOSS.ID_LOSS_TYPE = 'cross_entropy'
# output tensor index
_C.LOSS.ID_LOSS_OUTPUT_INDEX = 0

# name, weight, output_index 
# If train with label smooth, options: 'on', 'off'
_C.LOSS.COMPONENTS = [
    {"type": "cross_entropy", "weight": 1.0, "output_index": 0, "label_smooth": "off"},
    {"type": "triplet", "weight": 1.0, "output_index": 1, "margin": None},
    {"type": "center", "weight": 0.0005, "output_index": 1},
]

# ---------------------------------------------------------------------------- #
# PROCESSOR
# ---------------------------------------------------------------------------- #
_C.PROCESSOR = CN()
_C.PROCESSOR.TARGET_KEY = 1
_C.PROCESSOR.IMAGE_KEY = 0

# ---------------------------------------------------------------------------- #
# TEST
# ---------------------------------------------------------------------------- #

_C.TEST = CN()
# Number of images per batch during test
_C.TEST.IMS_PER_BATCH = 128
# If test with re-ranking, options: 'True','False'
_C.TEST.RE_RANKING = False
# Path to trained model
_C.TEST.WEIGHT = ""
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.FEAT_NORM = 'yes'

# Name for saving the distmat after testing.
_C.TEST.DIST_MAT = "dist_mat.npy"
# Whether calculate the eval score option: 'True', 'False'
_C.TEST.EVAL = False
#
_C.TEST.IMG_TEST_BATCH = 512
_C.TEST.TEST_BATCH = 32
_C.TEST.VIS = False
# ---------------------------------------------------------------------------- #
#Logging
# ---------------------------------------------------------------------------- #
_C.LOGGING = CN()
# Whether to use wandb
_C.LOGGING.WANDB_USE = False
# Project name in wandb
_C.LOGGING.WANDB_PROJECT = "reid"
# Experiment name in wandb
_C.LOGGING.WANDB_NAME = "baseline"
# Run id in wandb
_C.LOGGING.WANDB_RUN_ID = "00000"
# Whether to use tensorboard
_C.LOGGING.TENSORBOARD_USE = True


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = ""
_C.EXPERIMENT_NAME = "baseline"