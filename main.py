from utils.dataset import DataLoaderTorch
from utils.utils import attar_print, Cp
import torch
from modules.model import TModel, Encoder, Decoder
import os
import torchvision.transforms.functional as fn
import math
import torchvision.transforms as transforms
import time
import torch.nn as nn
import matplotlib.pyplot as plt


class HandiModule(nn.Module):
    def __init__(self):
        super(HandiModule, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
    
        
        
        encoder = self.encoder(x)
     
        x = self.decoder(encoder)
     
        return x


if __name__ == "__main__":
    print('Somethings...')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    epochs = 1000
    img_size = 28
    dtt = DataLoaderTorch(dataset="MNIST")
    # model = TModel(cfg='cfg/AEA-S.yaml').to(device=device)
    model = HandiModule().to(device)
    attar_print('Loading DataLoaderTorch Eval ...')
    data_eval = dtt.return_dataLoader_eval()
    attar_print('Done Loading DataLoaderTorch Eval [*]')
    attar_print('Loading DataLoaderTorch Train ...')
    data_train = dtt.return_dataLoader_train()
    attar_print('Done Loading DataLoaderTorch Train [*]')

    # print(hasattr(model,'forward'))
    iter_data_train = iter(data_train)
    iter_data_eval = iter(data_eval)
    batch_loop_train = math.ceil(data_train.__len__()/dtt.batch_size)
    batch_loop_eval = math.ceil(data_eval.__len__()/dtt.batch_size)
    loss_func = torch.nn.MSELoss()

    # optimizer = torch.optim.Adam(
    #     (list(model.encoder.parameters())+list(model.decoder.parameters())), 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3,weight_decay=1e-5)
    attar_print(
        f'tottal loops for train and eval the model and data are {batch_loop_train} / {batch_loop_eval}')
    for epoch in range(epochs):
        outputs = torch.zeros((32, 1, img_size, img_size))
        iter_data_train = iter(data_train)
        iter_data_eval = iter(data_eval)
        org = torch.zeros((32, 1, img_size, img_size))
        for i in range(batch_loop_train):
            s = time.time()
            try:
                data_train_batch = iter_data_train.next()
                optimizer.zero_grad()
                x, _ = data_train_batch
                x = fn.resize(x, size=[img_size, img_size])
                # x = fn.normalize(x,(0.5),(0.5))
                x = transforms.Normalize((0.5), (0.5))(x)
                x = x.to(device)
                y = model.forward(x)
                outputs = y
                org = x
                loss = loss_func(x, y)
                loss.backward()
                optimizer.step()
                attar_print(
                    f'\r epoch : {epoch} / {epochs} batch num : {i}/{batch_loop_train}   loss : {loss.item()} MAP : {time.time() - s}', end='', color=Cp.BLUE)
            except StopIteration:
                # iter_data_train = iter(data_train)
                attar_print('We Got Break Point \n')
                pass
                
           
        if epoch % 15 == 0:
            # plt.figure(figsize=(9,2))
            xp, yp = 5, 2
            fig, axes = plt.subplots(xp, yp)
            if not os.path.exists(f'results'):
                os.mkdir('results')
            plt.gray()
            for i in range(xp):
                # for j in range(yp):
                axes[i][0].imshow(
                    org[i].reshape(-1, img_size, img_size)[0].cpu().detach().numpy())
                axes[i][1].imshow(
                    outputs[i].reshape(-1, img_size, img_size)[0].cpu().detach().numpy())
            plt.savefig(f'results/{epoch}.png')
            # plt.show()
            if not os.path.exists(f'runs'):
                os.mkdir('runs')
            attar_print(f'\n\nSaving model ...', color=Cp.YELLOW)
            data_pack = {
                'model': model.state_dict(),
                # 'decoder': model.decoder.state_dict(),
                # 'encoder': model.encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(data_pack, f'runs/model-{epoch}-{loss.item():.4f}.pt')
            attar_print(
                f'Done Saving model continue [*] \n Local : runs/model-{epoch}-{loss.item():.4f}.pt', color=Cp.GREEN)
        print('')
