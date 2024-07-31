import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CLASSES_NUMBER

class Captcha_Model(nn.Module):
    def __init__(self):
        super(Captcha_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        self._calculate_flattened_size()
        
        self.fc1_bbox = nn.Linear(self.flattened_size, 128)
        self.fc2_bbox = nn.Linear(128, 4)
        
        self.fc1_class = nn.Linear(self.flattened_size, 128)
        self.fc2_class = nn.Linear(128, CLASSES_NUMBER)
    
    def _calculate_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 60, 210)
            x = self.conv1(dummy_input)
            x = self.pool(x)
            x = self.conv2(x)
            x = self.pool(x)
            self.flattened_size = x.view(-1).size(0)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        bbox = F.relu(self.fc1_bbox(x))
        bbox = self.fc2_bbox(bbox)
        
        class_scores = F.relu(self.fc1_class(x))
        class_scores = self.fc2_class(class_scores)
        print(bbox)
        return bbox, class_scores
