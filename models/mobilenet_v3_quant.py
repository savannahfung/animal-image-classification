import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.base_class import ImageClassificationBase, accuracy

# Quantizable MobileNetV3 Large Model
class QuantizableMobileNetV3(ImageClassificationBase):
    def __init__(self, num_classes, freeze_layers=True):
        super(QuantizableMobileNetV3, self).__init__()
        self.model = models.quantization.mobilenet_v3_large(pretrained=True, quantize=False)
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        for param in self.model.features[-8:].parameters():
            param.requires_grad = True
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x