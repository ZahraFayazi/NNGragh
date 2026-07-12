import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=10)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        fc1 = self.fc1(x)
        relu1 = self.relu1(fc1)
        drop1 = self.drop1(relu1)
        fc2 = self.fc2(drop1)
        relu2 = self.relu2(fc2)
        fc3 = self.fc3(relu2)
        out = self.out(fc3)
        return out

if __name__ == '__main__':
    device = torch.device('cuda')
    model = MLP().to(device)
    x = torch.randn(64, 784).to(device)
    print(model(x).shape)