## 공용 데이터 로더
from __future__ import print_function

import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import math
import sys
import random
from PIL import Image

from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np

from .data_transform import get_data_transform

from data.avmnist_dataloader import build_avmnist_image_loader, build_avmnist_loader, build_avmnist_sound_loader


def build_data_loader(args):
    if args.dataset == 'avmnist':
        return build_avmnist_loader(args)
    elif args.dataset == 'avmnist_image':
        return build_avmnist_image_loader(args)
    elif args.dataset == 'avmnist_sound':
        return build_avmnist_sound_loader(args)
    else:
        raise NotImplementedError
    
