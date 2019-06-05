import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 1, 1)
        # self.conv2 = nn.Conv2d(16, 16, 1, 1)
        #
        # self.conv3 = nn.Conv2d(16, 16, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 1, 1)

        self.conv5 = nn.Conv2d(16, 32, 1, 1)
        # self.conv6 = nn.Conv2d(32, 32, 1, 1)
        # self.conv7 = nn.Conv2d(32, 32, 1, 1)

        # self.conv8 = nn.Conv2d(32, 32, 1, 1)
        # self.conv9 = nn.Conv2d(32, 32, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 1, 1)

        # self.conv11 = nn.Conv2d(32, 64, 1, 1)
        # self.conv12 = nn.Conv2d(64, 64, 1, 1)
        self.conv13 = nn.Conv2d(64, 64, 1, 1)
        # self.fc1 = nn.Linear(64, 64)
        # self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv10(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv13(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc3(x))
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv5(x))
        # x = F.relu(self.conv6(x))
        # x = F.relu(self.conv7(x))
        # x = F.max_pool2d(x,2,2)
        # x = F.relu(self.conv8(x))
        # x = F.relu(self.conv9(x))
        # x = F.relu(self.conv10(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = F.relu(self.conv11(x))
        # x = F.relu(self.conv12(x))
        # x = F.relu(self.conv13(x))
        # x = F.max_pool2d(x, 2, 2)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        return x


# class VGG(nn.Module):
#     def __init__(self, input_channels=3, num_classes=10):
#         super(VGG, self).__init__()
#         self.block1 = self._make_layer(input_channels, 16, 2)
#         self.block2 = self._make_layer(16, 16, 2)
#         self.block3 = self._make_layer(16, 32, 3)
#         self.block4 = self._make_layer(32, 32, 3)
#         self.block5 = self._make_layer(32, 64, 3)
#         self.fc1 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         x = F.max_pool2d(self.block1(x), 2, 2)
#         x = F.max_pool2d(self.block2(x), 2, 2)
#         x = F.max_pool2d(self.block3(x), 2, 2)
#         x = F.max_pool2d(self.block4(x), 2, 2)
#         x = F.max_pool2d(self.block5(x), 2, 2)
#         x = x.view(x.size(0),-1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return x
#
#     def _make_layer(self, input_channels, output_channels, repeat):
#         layers = []
#         layers.append(ConvReLU(input_channels, output_channels))
#         for i in range(1, repeat):
#             layers.append(ConvReLU(output_channels, output_channels))
#         return nn.Sequential(*layers)


class ConvReLU(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 1, 1)

    def forward(self, x):
        return F.relu(self.conv(x))
