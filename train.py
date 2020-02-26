import torch
from model import model
# from torch.backends import cudnn
import numpy as np
import random
from utils.utils import load_frames

# random seed initialisation
manualSeed = 2020
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)

davis = r'/home/yiqian/Documents/dataset/DAVIS/JPEGImages/480p/scooter-gray'

# initialize the model
frames, frames_gray = load_frames(path=davis)
model = model(frames, frames_gray)
model.run_propagation(target=0, mode='naive', at_least=-1)




pass






















