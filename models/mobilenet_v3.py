import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.base_class import ImageClassificationBase, accuracy

# MobileNetV3 Large Model
class MobileNetV3(ImageClassificationBase):
    def __init__(self, num_classes, freeze_layers=True):
        super(MobileNetV3, self).__init__()
        self.model = models.mobilenet_v3_large(pretrained=True)
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.model.features[-6:].parameters():
            param.requires_grad = True
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)