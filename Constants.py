import sys
import os

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_ORIGIN = '/Users/polaras/Documents/Useful/dataset/data_original/'
DATA_TEMP = os.path.join(ROOT, 'data_temp/')

# Signal processing
SAMPLE_RATE = 16000
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
BUCKET_STEP = 1
MAX_SEC = 10

# Model
WEIGHTS_FILE = "data/model/weights.h5"
COST_METRIC = "cosine"  # euclidean or cosine
INPUT_SHAPE = (NUM_FFT, None, 1)
