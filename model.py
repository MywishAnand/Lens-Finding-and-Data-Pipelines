import torch
import torch.nn as nn
import torchvision.models as models

class LensClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(LensClassifier, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        
        # Modify the fully connected layer for binary classification output.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)
        
    def forward(self, x):
        return self.model(x).squeeze(1)
