from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_path = None):
    path = os.path.join(os.getcwd(), data_path) if data_path else os.getcwd()
    dataset = MNIST(path, 
                    download=True, 
                    transform=transforms.ToTensor())
    dataloader = DataLoader(dataset)
    return dataloader

