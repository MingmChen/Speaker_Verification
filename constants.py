import os

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_TEMP = os.path.join(ROOT, 'data_temp/')
DATA_TEMP = os.path.join(ROOT, 'data_temp/')

# DATA_ORIGIN = '/home/polaras/dataset/data_original/'
# DATA_ORIGIN = '/Users/polaras/Documents/Useful/dataset/data_original/'
DATA_ORIGIN = os.path.join(ROOT, 'data_temp_small/')

# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_COEF = 40
NUM_FFT = 512

# Train split
VALIDATION_SPLIT = 0.2
SHUFFLE_DATA = True

# Model training parameters
MODEL_DIR = ROOT + '/Models'
EPOCHS_PER_SAVE = 50
BATCH_SIZE = 32
BATCH_PER_LOG = 1
N_EPOCHS = 100

# Optimizer settings
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Scheduler setting
STEP_SIZE = 450
GAMMA = 0.1

CUBE_SHAPE = (80, 40, 20)
NUM_FILES = 0
