import torch
from torch import nn
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.shortcut = nn.Sequential()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv_input = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, input):
        #shortcut = self.shortcut(input)
        shortcut = self.conv_input(input)
        shortcut= self.bn3(shortcut)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = (self.bn2(self.conv2(input)))
        # get dimensions of the output (+ heigth and width of image as stride) kernal size size one
        input = nn.ReLU()(input)
        input = input + shortcut
        return input

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(64, 64, 1)
        )

        self.layer2 = nn.Sequential(
            ResBlock(64, 128,2)
        )

        self.layer3 = nn.Sequential(
            ResBlock(128, 256, 2)
        )


        self.layer4 = nn.Sequential(
            ResBlock(256, 512, 2)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(512, 2)

    def forward(self, input):

        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input,start_dim=1)
        input = self.fc(input)
        input=  torch.sigmoid(input)
        #print(input)
        return input