import os
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision import transforms
from data import train_val_dataset

class constant():
    def __init__(self, dir):
        self._data_dir = dir
        self._data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        self._image_datasets = train_val_dataset(ImageFolder(self._data_dir, transform=Compose([Resize((224,224)),ToTensor()])))
        self._dataloader ={x: torch.utils.data.DataLoader(self._image_datasets[x], batch_size=256, shuffle=True) for x in ['train', 'val']}
        self._dataset_sizes = {x: len(self._image_datasets[x]) for x in ['train', 'val']}
        self._class_names = self._image_datasets['train'].dataset.classes
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._dataset = dataset = ImageFolder(self._data_dir, transform=Compose([Resize((224,224)),ToTensor()]))
        
    @property
    def data_dir(self):
        return self._data_dir
    @property
    def data_transform(self):
        return self._data_transform 
    @property
    def image_datasets(self):
        return self._image_datasets
    @property
    def dataloader(self):
        return self._dataloader
    @property
    def dataset_sizes(self):
        return self._dataset_sizes
    @property
    def class_names(self):
        return self._class_names
    @property
    def device(self):
        return self._device
    @property
    def dataset(self):
        return self._dataset