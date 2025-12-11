"""
PyTorch CNN models for Alzheimer's MRI Classification
Intermediate-level models: Custom CNN and ResNet50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AlzheimersCNN(nn.Module):
    """
    Custom CNN model for Alzheimer's classification
    Architecture: Conv layers with batch norm -> FC layers
    """
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(AlzheimersCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Store for Grad-CAM
        self.layer4 = self.features[-5:-1]
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResNetModel(nn.Module):
    """
    ResNet50-based model for Alzheimer's classification
    Using transfer learning with pretrained ImageNet weights
    """
    def __init__(self, pretrained=True, num_classes=4):
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet50
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer to accept 1 channel (grayscale)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify final fc layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Store layer4 for Grad-CAM
        self.layer4 = self.resnet.layer4
    
    def forward(self, x):
        return self.resnet(x)

class SimpleResNet(nn.Module):
    """
    Simpler ResNet variant for faster training
    """
    def __init__(self, pretrained=True, num_classes=4):
        super(SimpleResNet, self).__init__()
        
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        self.layer4 = self.resnet.layer4
    
    def forward(self, x):
        return self.resnet(x)

class EnsembleModel(nn.Module):
    """
    Ensemble of multiple models for improved predictions
    """
    def __init__(self, models_list, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models_list)
        
        if weights is None:
            self.weights = [1.0 / len(models_list)] * len(models_list)
        else:
            self.weights = weights
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        ensemble_output = sum(w * o for w, o in zip(self.weights, outputs))
        return ensemble_output

if __name__ == "__main__":
    # Test models
    batch_size = 4
    x = torch.randn(batch_size, 1, 224, 224)
    
    print("Testing Custom CNN...")
    cnn = AlzheimersCNN(num_classes=4)
    out_cnn = cnn(x)
    print(f"Output shape: {out_cnn.shape}")
    
    print("\nTesting ResNet50...")
    resnet = ResNetModel(pretrained=False, num_classes=4)
    out_resnet = resnet(x)
    print(f"Output shape: {out_resnet.shape}")
    
    print("\nTesting SimpleResNet...")
    simple_resnet = SimpleResNet(pretrained=False, num_classes=4)
    out_simple = simple_resnet(x)
    print(f"Output shape: {out_simple.shape}")
