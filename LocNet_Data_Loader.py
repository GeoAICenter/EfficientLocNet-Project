from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torch
import os


class RadioMapDataset(Dataset):
    def __init__(self, data_path, typeOfData, typeOfAntenna='antennas'):
        self.data_path = data_path
        self.typeOfData = typeOfData
        self.typeOfAntenna = typeOfAntenna
        self.list_images = os.listdir(self.data_path + '/' + typeOfData)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, idx):
        antenna = '_'.join(self.list_images[idx].split('_')[:2]) + '.png'
        single_signal_path = self.data_path + '/' + 'DPM' + '/' + antenna
        mask = self.data_path + '/' + self.typeOfData + '/' + self.list_images[idx]
        target = self.data_path + '/' + self.typeOfAntenna + '/' + antenna
        single_signal_image = read_image(single_signal_path, ImageReadMode.GRAY).float()
        mask = read_image(mask, ImageReadMode.GRAY).float()
        target = read_image(target, ImageReadMode.GRAY).float() / 255
        single_signal_image /= 255.0
        data = single_signal_image * mask
        return data, target
