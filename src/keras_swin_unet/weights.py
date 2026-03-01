import numpy as np
from PIL import Image

from keras_swin_unet.utils import dummy_loader

import sys
sys.path.append("/home/user_one/projects/GEOINT/keras-swin-unet/src")

dummy_loader("/home/user_one/projects/GEOINT/best_model.keras")