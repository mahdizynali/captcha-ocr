import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CLASSES_NUMBER, IMAGES_SIZE

class Captcha_OCR(nn.Module):
    def __init__(self, num_classes=CLASSES_NUMBER, max_digits_in_row=4):
        super(Captcha_OCR, self).__init__()

        self.max_digits = max_digits_in_row  
        self.num_classes = num_classes  

        self.backbone_location = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 13))  # Ensure consistent shape
        )

        # Calculate flattened feature size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, *IMAGES_SIZE)
            dummy_features = self.backbone_location(dummy_input)
            flattened_size = dummy_features.numel()

        # Bounding Box Regression Head
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4 * self.max_digits)
        )

        # Classification Head
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * self.max_digits)
        )

    def forward(self, x):
        features = self.backbone_location(x)

        bbox_pred = self.bbox_head(features)
        bbox_pred = bbox_pred.view(-1, self.max_digits, 4)

        class_pred = self.classification_head(features)
        class_pred = class_pred.view(-1, self.max_digits, self.num_classes)

        return bbox_pred, class_pred
