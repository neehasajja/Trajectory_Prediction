import torch
import torch.nn as nn
import torch.nn.init as init
from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50
from tqdm import tqdm


class EarlyStopping(object):
    def __init__(self, model, save_path, mode='min', min_delta=0, patience=10, percentage=False):
        
