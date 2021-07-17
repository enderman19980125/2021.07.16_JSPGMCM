import torch
import torchvision
import torch.nn.functional


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature = torchvision.models.resnet18(pretrained=True)
        self.fc = torch.nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        x = self.feature(x)
        x = torch.nn.functional.relu(x)
        x = self.fc(x)
        return x
