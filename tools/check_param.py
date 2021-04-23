import sys

import torchvision
sys.path.append("..")

from archs.repvgg import repvgg_A0, repvgg_B1g2
import torch
import torchvision

# model = getattr(torchvision.models, "resnet50")(True)

model = repvgg_A0(pretrained=False, deploy=False)

import pdb; pdb.set_trace()

checkpoint = torch.load("../pretrain_model/RepVGG-A0-train.pth")
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

import pdb; pdb.set_trace()

