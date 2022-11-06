from ast import Try
from multiprocessing.spawn import import_main_path
import os

import numpy as np
from sympy import comp
import torch
import yaml
import cv2 as cv
from torchvision import datasets
from .utils import Cp, attar_print
from torch.utils.data import DataLoader, Dataset
import threading
import torchvision.transforms as transforms
from .utils import read_yaml
from torchvision.transforms import *


def load_from_file(path):
    res = [os.path.join(path, v) for v in os.listdir(path) if os.path.exists(os.path.join(path, v))]
    return res


class CommonDataSet(Dataset):
    def __init__(self, x, y=None):
        self.x = x.float()
        del x
        self.y = y
        del y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        # print(f'y   = {self.y}')
        # print(f'x   = {self.x}')
        # return self.x[item] if self.y is None else self.x[item], self.y[item]

        if self.y is None:
            return self.x[item]
        else:
            return self.x[item], self.y[item]


class DataLoaderTorch:
    def __init__(self, dataset: str = None, batch_size: int = 32,
                 shuffle: bool = True, num_workers: int = 2,
                 transform=None):
        self.dataset = 'MNIST' if dataset is None else dataset
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_size = batch_size
        compose = transforms.ToTensor()

        self.transform = compose if transform is None else transform
        self.start()

    def start(self):

        if self.dataset is not None:

            print(f"{Cp.CYAN}Searching for {self.dataset}{Cp.RESET} ... \n", end='')
            if self.dataset.endswith('.mp4'):
                return NotImplementedError
            elif self.dataset.endswith('.yaml'):
                x = []
                yam = read_yaml(self.dataset)
                files = load_from_file(yam['image_path'])
                total_files = len(files)
                print(
                    f"{Cp.CYAN}Files Are Ready To Load for {self.dataset} {total_files} images are Found {Cp.RESET} ... \n",
                    end='')

                for i, file in enumerate(files):
                    img = cv.cvtColor(cv.imread(file), cv.COLOR_BGR2RGB)
                    shape = img.shape
                    # print(shape)
                    img = img.reshape(shape[::-1])
                    # print(f'shape  :  {img.shape}')
                    x.append(img)
                    attar_print(f'\rLoading Images : ', f'{i + 1}/{total_files}', end='')
                attar_print(f'\nLoading Images is Done [*] ', end='\n')
                x = CommonDataSet(x=torch.tensor(x), y=None)
                # x = torch.from_numpy(np.asarray(x))
                attar_print(f'\nLoading CommonDataSet is Done [*] ', end='\n', color=Cp.CYAN)
                self.data_train = x
                self.data_eval = None
            else:
                if os.path.exists(f'data/downloaded_data_tarin_{self.dataset}'):
                    print(
                        f"{Cp.CYAN}{self.dataset} Train Found Skip Download {self.dataset}{Cp.RESET} [*] \n", end='')
                    download_train = False
                else:
                    print(
                        f"{Cp.RED}{self.dataset} Train not Found Download Added To Theread {Cp.RESET} [*] \n", end='')
                    download_train = True

                if os.path.exists(f'data/downloaded_data_val_{self.dataset}'):
                    print(
                        f"{Cp.CYAN}{self.dataset} Val Found Skip Download {self.dataset}{Cp.RESET} [*] \n", end='')
                    download_val = False
                else:
                    print(
                        f"{Cp.RED}{self.dataset} Val not Found Download Added To Theread {Cp.RESET} [*] \n{Cp.BLUE}",
                        end='')
                    download_val = True

                self.data_train = eval(
                    f"datasets.{self.dataset}(root='data/downloaded_data_tarin_{self.dataset}',train=True,download={download_train},transform={self.transform})")
                self.data_eval = eval(
                    f"datasets.{self.dataset}(root='data/downloaded_data_val_{self.dataset}',train=False,download={download_val},transform={self.transform})")

                try:
                    thread = [threading.Thread(target=eval(self.data_train)), threading.Thread(
                        target=eval(self.data_eval))]
                    for t in thread:
                        t.start()
                except:
                    print(
                        f"{Cp.RED}Searching for {self.dataset} Faild Dataset NotFound {Cp.RESET} \n", end='')

    def return_dataLoader_train(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def return_dataLoader_eval(self):
        return DataLoader(self.data_eval, batch_size=self.batch_size,
                          shuffle=True) if self.data_eval is not None else None
