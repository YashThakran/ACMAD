import torch.nn as nn
import torch.nn.functional as F
import torch

class RSM1D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)

        self.nin = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)
        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx
    
class SSDNet1D(nn.Module):  # ResNet
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(100)
        self.RSM1 = RSM1D(channels_in=100, channels_out=256)
        self.RSM2 = RSM1D(channels_in=256, channels_out=256)
        self.RSM3 = RSM1D(channels_in=256, channels_out=512)
        self.RSM4 = RSM1D(channels_in=512, channels_out=512)
        self.fc1 = nn.Linear(in_features=512, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM4(x)
        x = F.max_pool1d(x, kernel_size=x.shape[-1])
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    
    
class DilatedCovModule(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        channels_out = int(channels_out/4)
        self.cv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=1, padding=1)
        self.cv2 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=2, padding=2)
        self.cv4 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=4, padding=4)
        self.cv8 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, dilation=8, padding=8)
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn4 = nn.BatchNorm1d(channels_out)
        self.bn8 = nn.BatchNorm1d(channels_out)

    def forward(self, xx):
        xx1 = F.relu(self.bn1(self.cv1(xx)))
        xx2 = F.relu(self.bn2(self.cv2(xx)))
        xx4 = F.relu(self.bn4(self.cv4(xx)))
        xx8 = F.relu(self.bn8(self.cv8(xx)))
        yy = torch.cat((xx1, xx2, xx4, xx8), dim=1)
        return yy


class DilatedNet(nn.Module):  # IncNet
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(8)

        self.DCM1 = DilatedCovModule(channels_in=8, channels_out=16)
        self.DCM2 = DilatedCovModule(channels_in=16, channels_out=32)
        self.DCM3 = DilatedCovModule(channels_in=32, channels_out=32)
        self.DCM4 = DilatedCovModule(channels_in=32, channels_out=64)

        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)
        self.out = nn.Linear(in_features=10, out_features=2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)
        x = F.max_pool1d(self.DCM1(x),kernel_size=4)
        x = F.max_pool1d(self.DCM2(x), kernel_size=4)
        x = F.max_pool1d(self.DCM3(x), kernel_size=4)
        x = F.max_pool1d(self.DCM4(x), kernel_size=x.shape[-1])
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
