import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

def create_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = MNIST(root='./mnistdata', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./mnistdata', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_train_loader(batch_size):
    train_loader, _ = create_data_loaders(batch_size)
    return train_loader

def create_test_loader(batch_size):
    _, test_loader = create_data_loaders(batch_size)
    return test_loader