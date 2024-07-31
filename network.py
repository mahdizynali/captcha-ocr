import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CLASSES_NUMBER

class Captcha_Model(nn.Module):

    def __init__(self, in_channels=1, num_classes=CLASSES_NUMBER, num_bboxes=4):
        super(Captcha_Model, self).__init__()
        
        # Define the convolutional layers
        hidden_channel1 = 32
        hidden_channel2 = 64
        hidden_channel3 = 128
        kernel_size = 3
        stride = 1
        padding = 1

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=hidden_channel1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=hidden_channel1, out_channels=hidden_channel2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv2d(in_channels=hidden_channel2, out_channels=hidden_channel2, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv2d(in_channels=hidden_channel2, out_channels=hidden_channel3, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv5 = nn.Conv2d(in_channels=hidden_channel3, out_channels=hidden_channel3, kernel_size=kernel_size, stride=stride, padding=padding)

        self.batchnorm1 = nn.BatchNorm2d(hidden_channel1)
        self.batchnorm2 = nn.BatchNorm2d(hidden_channel2)
        self.batchnorm3 = nn.BatchNorm2d(hidden_channel3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc_input_size = self._get_fc_input_size()
        
        self.cnn_layer = nn.Linear(self.fc_input_size, num_classes)
        self.regressor = nn.Linear(self.fc_input_size, num_bboxes * 4)

    def _get_fc_input_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 60, 210)
            x = self.feature_extractor(dummy_input)
            x = self.flatten(x)
            return x.numel()

    def cnn_layers(self, x):
        x = self.relu(x)
        x = self.maxpool(x)
        return x
    
    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.cnn_layers(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.cnn_layers(x)
        
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.cnn_layers(x)
        
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = self.cnn_layers(x)
        
        x = self.conv5(x)
        x = self.batchnorm3(x)
        x = self.cnn_layers(x)
        
        return x
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        classifier_op = self.cnn_layer(x)
        regressor_op = self.regressor(x)
        return (regressor_op, classifier_op)

