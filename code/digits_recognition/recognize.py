import torch
import torchvision
import torch.utils.data
import numpy as np
from .Net import Net


def recognize_digit(image: np.ndarray) -> int:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda tensor: tensor.repeat(1, 3, 1, 1))
    ])

    net = Net().eval().cuda()
    net.load_state_dict(torch.load("digits_recognition/4.pth"))

    data = transform(image).cuda()
    y_out = net(data)
    y_out = y_out.cpu()
    _, y_pred = torch.max(y_out, dim=1)

    return int(y_pred)
