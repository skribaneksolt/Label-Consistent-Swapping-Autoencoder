import models.networks.network_deeplab as network_deeplab
import models.networks.utils_deeplab as utils_deeplab
import torch
import torch.nn as nn
from pathlib import Path


def create_outer_segmenter(num_classes=16, output_stride=16,
                           checkpoint_path=Path().resolve().parent.parent):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
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
