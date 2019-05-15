import os

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_TEMP = os.path.join(ROOT, 'data_temp/')

DATA_ORIGIN = '/Users/polaras/Documents/Useful/dataset/data_original/'


# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_COEF = 40
NUM_FFT = 1024
BUCKET_STEP = 1
MAX_SEC = 10


# Model training
MODEL_DIR = ROOT + '/Models'
EPOCHS_PER_SAVE = 20