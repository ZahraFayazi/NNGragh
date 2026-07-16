import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    def __init__(self):
        super(InceptionBlock, self).__init__()
        # Branch A
        self.convA1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.reluA = nn.ReLU()
        # Branch B
        self.convB1 = nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0)
        self.convB2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.reluB = nn.ReLU()
        # Merge
        self.bn1 = nn.BatchNorm2d(48)
        self.out = nn.ReLU()

    def forward(self, x):
        # Branch A
        a = self.convA1(x)
        a = self.reluA(a)
        # Branch B
        b = self.convB1(x)
        b = self.convB2(b)
        b = self.reluB(b)
        # Concat and finalize
        x = torch.cat([a, b], dim=1)  # shape: (batch, 48, 56, 56)
        x = self.bn1(x)
        x = self.out(x)
        return x

if __name__ == '__main__':
    device = torch.device('cpu')
    model = InceptionBlock().to(device)
    x = torch.randn(1, 32, 56, 56).to(device)
    print(model(x).shape)
