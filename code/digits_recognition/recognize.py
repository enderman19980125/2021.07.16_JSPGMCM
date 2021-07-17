import os
import cv2
import torch
import torchvision
import torch.utils.data
import numpy as np
from tqdm import tqdm
from Net import Net


def recognize_digit(image: np.ndarray) -> int:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda tensor: tensor.repeat(1, 3, 1, 1))
    ])

    data = transform(image).cuda()
    y_out = net(data)
    y_out = y_out.cpu()
    _, y_pred = torch.max(y_out, dim=1)

    return int(y_pred)


def move_digit_images() -> None:
    num_train = [0] * 10
    num_test = [0] * 10

    for digit_image in tqdm(os.listdir("../../data/digits/u")):
        src_path = os.path.join("../../data/digits/u", digit_image)
        image = cv2.imread(src_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        digit = recognize_digit(image)
        if np.random.random() < 0.8:
            if num_train[digit] >= 1000:
                break
            dst_path = os.path.join(f"../../data/digits/train/{digit}", f"{num_train[digit]}.jpg")
            num_train[digit] += 1
        else:
            if num_test[digit] >= 1000:
                break
            dst_path = os.path.join(f"../../data/digits/test/{digit}", f"{num_test[digit]}.jpg")
            num_test[digit] += 1
        os.rename(src_path, dst_path)


if __name__ == '__main__':
    net = Net().eval().cuda()
    net.load_state_dict(torch.load("mnist_5e.pth"))
    move_digit_images()
