from utils.dataset import DataLoaderTorch
from utils.utils import attar_print, Cp
import torch
from modules.model import TModel, Encoder, Decoder
import os
import torchvision.transforms.functional as fn
import math
import torchvision.transforms as transforms
import time
from utils.train import train
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dtp = DataLoaderTorch(batch_size=1, dataset='data/data.yaml')
    # img_size = (width, height)
    train(5000, 4, dtp, img_size=(1920, 1040),prp=20)
