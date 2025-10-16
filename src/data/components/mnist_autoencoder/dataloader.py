from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os

def get_dataloaders():
    path = os.path.join(os.getcwd(), 'data')
    dataset = MNIST(path, 
                    download=True, 
                    transform=transforms.ToTensor())
    dataloader = DataLoader(dataset)
    return dataloader

