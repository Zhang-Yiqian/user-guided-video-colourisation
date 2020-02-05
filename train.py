import torch
import model
from torch.backends import cudnn
import numpy as np
import random
from utils import load_frames

# random seed initialisation
manualSeed = 2020
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)

davis = '~/Documents/dataset/DAVIS/JPEGImages/480p/scooter-gray/'

# initialize the model
frames = load_frames(davis)
model = model(frames)


















