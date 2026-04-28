import torch.nn as nn
from torchvision import models

def get_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True