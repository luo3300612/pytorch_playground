import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.block1 = self._make_layer(input_channels, 64, 2)
        self.block2 = self._make_layer(64, 128, 2)
        self.block3 = self._make_layer(128, 256, 3)
        self.block4 = self._make_layer(256, 512, 3)
        self.block5 = self._make_layer(512, 512, 3)
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _make_layer(self, input_channels, output_channels, repeat, batch_norm=True):
        layers = []
        layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, repeat):
            layers.append(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(output_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
