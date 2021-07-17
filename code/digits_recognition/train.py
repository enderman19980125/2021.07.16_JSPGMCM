import torch
import torchvision
import torch.utils.data
from tqdm import tqdm
from sklearn import metrics
from Net import Net


def train(image_folder: str, num_epochs: int = 100, batch_size: int = 32, lr: float = 0.001, load_pth: str = None, save_pth: str = None):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda tensor: tensor.repeat(3, 1, 1))
    ])
    train_data = torchvision.datasets.MNIST(root=image_folder, transform=transform, train=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)

    net = Net()
    if torch.cuda.is_available():
        net = Net().cuda()
        print("Cuda is available.")
    else:
        print("Cuda is unavailable.")

    if load_pth:
        net.load_state_dict(torch.load(load_pth))
        print(f'Loaded "{load_pth}".')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch_id in range(num_epochs):
        print(f'----------------    Epoch {epoch_id}    ----------------')
        y_pred_list = []
        y_true_list = []

        for data, y_true in tqdm(train_loader):
            data = data.cuda() if torch.cuda.is_available() else data
            y_true = y_true.cuda() if torch.cuda.is_available() else y_true

            y_out = net(data)
            loss = criterion(y_out, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_out = y_out.cpu()
            _, y_pred = torch.max(y_out, dim=1)
            y_pred_list.extend(y_pred.tolist())
            y_true_list.extend(y_true.tolist())

        accuracy = metrics.accuracy_score(y_true=y_true_list, y_pred=y_pred_list)
        print(accuracy)
        torch.save(net.state_dict(), f"{epoch_id}.pth")

    if save_pth:
        torch.save(net.state_dict(), save_pth)
        print(f'Saved "{save_pth}".')


if __name__ == '__main__':
    train(image_folder="./../../../DataSets")
