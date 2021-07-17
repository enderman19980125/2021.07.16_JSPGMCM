import torch
import torchvision
import torch.utils.data
from tqdm import tqdm
from sklearn import metrics
from Net import Net


def test(image_folder: str, batch_size: int = 32, load_pth: str = None):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda tensor: tensor.repeat(3, 1, 1))
    ])
    test_data = torchvision.datasets.MNIST(root=image_folder, transform=transform, train=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

    net = Net()
    net.eval()
    if torch.cuda.is_available():
        print("Cuda is available.")
        net = Net().cuda()
    else:
        print("Cuda is unavailable.")

    if load_pth:
        net.load_state_dict(torch.load(load_pth))
        print(f'Loaded "{load_pth}".')

    y_pred_list = []
    y_true_list = []

    for data, y_true in tqdm(test_loader):
        data = data.cuda() if torch.cuda.is_available() else data
        y_true = y_true.cuda() if torch.cuda.is_available() else y_true

        y_out = net(data)

        y_out = y_out.cpu()
        _, y_pred = torch.max(y_out, dim=1)
        y_pred_list.extend(y_pred.tolist())
        y_true_list.extend(y_true.tolist())

    accuracy = metrics.accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
    print(accuracy)


if __name__ == '__main__':
    test(image_folder="./../../../DataSets", load_pth="4.pth")
