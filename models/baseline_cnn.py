import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_class import ImageClassificationBase, accuracy

# Baseline CNN
class BaselineCNN(ImageClassificationBase):
    def __init__(self, classes):
        super().__init__()
        self.num_classes=classes
        self.conv1=nn.Conv2d(3, 64, 5, 1)
        self.conv2=nn.Conv2d(64, 128, 3, 1)
        self.conv3=nn.Conv2d(128, 128, 3, 1)
        self.conv4=nn.Conv2d(128, 128, 3, 1)
        self.fc1=nn.Linear(128 * 5 * 5, self.num_classes)
        
    def forward(self,X):
        X=F.relu(self.conv1(X))
        X=F.max_pool2d(X, 2, 2)
        X=F.relu(self.conv2(X))
        X=F.max_pool2d(X, 2, 2)
        X=F.relu(self.conv3(X))
        X=F.max_pool2d(X, 2, 2)
        X=F.relu(self.conv4(X))
        X=F.max_pool2d(X, 2, 2)
        X=X.view(-1, 128 * 5 * 5)
        X=self.fc1(X)

        return F.log_softmax(X, dim=1)