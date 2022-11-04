from ast import Try
from multiprocessing.spawn import import_main_path
import os
from sympy import comp
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from .utils import Cp,attar_print
import threading
import torchvision.transforms as transforms
from torchvision.transforms import *

class DataLoaderTorch:
    def __init__(self, dataset: str = None, batch_size: int = 32,
     shuffle: bool = True, num_workers: int = 2,
     transform=None):
        self.dataset = 'MNIST' if dataset is not None else dataset
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.batch_size = batch_size
        compose =  transforms.ToTensor()
        
        self.transform =  compose if transform is None else transform
        self.start()

    def start(self):

        if self.dataset is not None:
            print(f"{Cp.CYAN}Searching for {self.dataset}{Cp.RESET} ... \n", end='')
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
                    f"{Cp.RED}{self.dataset} Val not Found Download Added To Theread {Cp.RESET} [*] \n{Cp.BLUE}", end='')
                download_val = True
            
            
            self.data_train = eval(
                f"datasets.{self.dataset}(root='data/downloaded_data_tarin_{self.dataset}',train=True,download={download_train},transform={self.transform})")
            self.data_eval = eval(
                f"datasets.{self.dataset}(root='data/downloaded_data_val_{self.dataset}',train=False,download={download_val},transform={self.transform})")
            
      

            try:
                thered = [threading.Thread(target=eval(self.data_train)), threading.Thread(
                    target=eval(self.data_eval))]
                for t in thered:
                    t.start()
            except:
                print(
                    f"{Cp.RED}Searching for {self.dataset} Faild Dataset NotFound {Cp.RESET} \n", end='')
        else:
            if not self.dataset.enddwith('.yaml'):
                raise 'Excepted data.yaml file to load data from but got {}'.format(
                    self.dataset)

    def return_dataLoader_train(self):
        return DataLoader(self.data_train,batch_size=self.batch_size,shuffle=True)

    def return_dataLoader_eval(self):
        return DataLoader(self.data_eval,batch_size=self.batch_size,shuffle=True)
