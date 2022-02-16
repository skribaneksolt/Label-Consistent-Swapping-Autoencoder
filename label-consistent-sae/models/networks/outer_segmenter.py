# from torch.utils.data import dataset
# from tqdm import tqdm
import models.networks.network_deeplab as network_deeplab
import models.networks.utils_deeplab as utils_deeplab
# import os
# import random
# import argparse
# import numpy as np
#
# from torch.utils import data
# # from datasets import VOCSegmentation, Cityscapes, cityscapes
# from torchvision import transforms as T
# from models.networks.metrics_deeplab import StreamSegMetrics

import torch
import torch.nn as nn

# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt
# from glob import glob

def create_outer_segmenter(num_classes=16, output_stride=16, checkpoint_path='/home/general/swapping-inout/out_segmenter.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # NUM_CLASSES = 18
    # OUTPUT_STRIDE = 16
    # checkpoint_path = ''
    # model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    model = network_deeplab.deeplabv3plus_mobilenet(num_classes, output_stride)
    utils_deeplab.set_bn_momentum(model.backbone, momentum=0.01)
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    # model.eval()
    for parameter in model.parameters(recurse=True):
        parameter.requires_grad = False
    print("Resume model from %s" % checkpoint_path)
    del checkpoint
    return model
