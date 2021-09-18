import torch.nn as nn, torch, torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        """
        网络的各层具体结构定义
        """
        self.layer1 = nn.Sequential(nn.BatchNorm1d(2),
                                    nn.Conv1d(2, 16, 21, 1, padding=10),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Conv1d(16, 16, 21, 1, padding=10),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(16, 16, 21, 1, padding=10),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv1d(16, 24, 11, 1, padding=5),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU(inplace=True),
                                    nn.Conv1d(24, 24, 11, 1, padding=5),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(24, 24, 11, 1, padding=5),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv1d(24, 48, 5, 1, padding=2),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU(inplace=True),
                                    nn.Conv1d(48, 48, 5, 1, padding=2),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(48, 48, 5, 1, padding=2),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU(inplace=True))
        self.layer4_1 = nn.Sequential(nn.Conv1d(48, 96, 5, 1, padding=2),
                                      nn.BatchNorm1d(96),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.15),
                                      nn.Conv1d(96, 96, 5, 1, padding=2),
                                      nn.BatchNorm1d(96),
                                      nn.ReLU(inplace=True))
        self.layer4_2 = nn.Sequential(nn.Conv1d(48, 96, 5, 1, padding=20, dilation=10),
                                      nn.BatchNorm1d(96),
                                      nn.ReLU(inplace=True),
                                      nn.Dropout(0.15),
                                      nn.Conv1d(96, 96, 5, 1, padding=20, dilation=10),
                                      nn.BatchNorm1d(96),
                                      nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Conv1d(336, 48, 5, 1, padding=2),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(48, 48, 5, 1, padding=2),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(48, 48, 5, 1, padding=2),
                                    nn.BatchNorm1d(48),
                                    nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Conv1d(72, 24, 11, 1, padding=5),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(24, 24, 11, 1, padding=5),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(24, 24, 11, 1, padding=5),
                                    nn.BatchNorm1d(24),
                                    nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Conv1d(40, 16, 21, 1, padding=10),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(16, 16, 21, 1, padding=10),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(0.15),
                                    nn.Conv1d(16, 16, 21, 1, padding=10),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(inplace=True))
        self.layer8 = nn.Conv1d(16, 1, 1, 1, padding=0)

    def forward(self, x):
        """
        网络各层的输入输出传递
        """
        x1 = self.layer1(x)
        p1 = F.max_pool1d(x1, 10)
        x2 = self.layer2(p1)
        p2 = F.max_pool1d(x2, 5)
        x3 = self.layer3(p2)
        p3 = F.max_pool1d(x3, 2)
        x4_1 = self.layer4_1(p3)
        x4_2 = self.layer4_2(p3)
        x4_3 = x4_1 - x4_2
        x4 = torch.cat([x4_1, x4_2, x4_3], dim=1)
        # temp1 = F.upsample(x4, scale_factor=2)
        temp1 = F.interpolate(x4, scale_factor=2)
        merge1 = torch.cat([temp1, x3], dim=1)
        x5 = self.layer5(merge1)
        # temp2 = F.upsample(x5, scale_factor=5)
        temp2 = F.interpolate(x5, scale_factor=5)
        merge2 = torch.cat([temp2, x2], dim=1)
        x6 = self.layer6(merge2)
        # temp3 = F.upsample(x6, scale_factor=10)
        temp3 = F.interpolate(x6, scale_factor=10)
        merge3 = torch.cat([temp3, x1], dim=1)
        x7 = self.layer7(merge3)
        x8 = self.layer8(x7)
        x9 = torch.sigmoid(x8)
        return x9
