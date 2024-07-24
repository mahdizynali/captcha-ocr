import torch
from config import *
import torch.nn as nn
import torch.nn.functional as F

class Captcha_Model(nn.Module):
    def __init__(self):
        super(Captcha_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate the size of the flattened tensor
        self._initialize_weights()
        self._calculate_flattened_size()
        
        # Define fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, CLASSES_NUMBER)
    
    def _initialize_weights(self):
        # Initializing weights to ensure the network is set up
        # This step can be omitted if default initialization is sufficient
        pass
    
    def _calculate_flattened_size(self):
        # Pass a dummy input through the network to compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 60, 210)  # Single sample, 1 channel, 60x210
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
        x = x.view(-1, self.flattened_size)  # Flatten based on computed size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
