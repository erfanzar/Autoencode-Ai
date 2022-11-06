from .dataset import DataLoaderTorch
from .utils import attar_print, Cp
import torch
from modules.model import TModel, Encoder, Decoder
import os
import torchvision.transforms.functional as fn
import math
import torchvision.transforms as transforms
import time
import torch.nn as nn
import matplotlib.pyplot as plt


def forward(data_train, epochs, batch_size, img_size, batch_loop_train, optimizer, model, loss_func, c_epoch, prp,
            device):
    height = img_size[1]
    width = img_size[0]
    outputs = torch.zeros((batch_size, 3, width, height))
    iter_data_train = iter(data_train)

    org = torch.zeros((batch_size, 3, width, height))
    d = []
    for i in range(batch_loop_train):
        s = time.time()
        try:
            data_train_batch = iter_data_train.next()
            optimizer.zero_grad()
            x = data_train_batch

            x = fn.resize(x, size=img_size)
            x = x.to(device)
            y = model.forward(x)
            outputs = y
            org = x
            loss = loss_func(x, y)
            loss *= batch_size
            loss.backward()
            optimizer.step()
            d.append(loss.item())
            attar_print(
                '\r {:>8}:{:>6}/{:>6}  {:>9}:{:>5}/{:>5}  {:>10}:{:>20}  {:>5}:{:>25} '.format('epoch', c_epoch, epochs,
                                                                                               "batch num", i,
                                                                                               batch_loop_train, 'Loss',
                                                                                               loss.item(), "MAP",
                                                                                               time.time() - s),
                end='', color=Cp.BLUE)
        except StopIteration:
            attar_print('We Got Break Point \n')
            pass

    if c_epoch % prp == 0:
        # plot the Results or model summery
        xp, yp = 2, 2
        fig, axes = plt.subplots(xp, yp)
        if not os.path.exists(f'results'):
            os.mkdir('results')
        # plt.gray()
        print(((fn.resize(org[i].reshape(1, 3, width, height)[0],
                            size=[width // 3, height // 3]).cpu().reshape(width // 3, height // 3,
                                                                                     3).detach() * 255) // 1).int().shape)
        for i in range(1):
            axes[i][0].imshow(

                ((fn.resize(org[i].reshape(1, 3, width, height)[0],
                            size=[width // 3, height // 3]).cpu().reshape(width // 3, height // 3,
                                                                                     3).detach() * 255) // 1).int().numpy())
            axes[i][0].xticks([]), axes[i][0].yticks([])
            axes[i][1].imshow(
                ((fn.resize(outputs[i].reshape(1, 3, width, height)[0],
                            size=[width // 3, height // 3]).cpu().reshape(width // 3, height // 3,
                                                                                     3).detach() * 255) // 1).int().numpy())
            axes[i][1].xticks([]), axes[i][1].yticks([])
        axes[1][0].semilogx(torch.tensor(d).cpu().detach().numpy() * 100)

        plt.savefig(f'results/{c_epoch}.png')
        # plt.show()
        if not os.path.exists(f'runs'):
            os.mkdir('runs')
        attar_print(f'\n\nSaving model ...', color=Cp.YELLOW)
        data_pack = {
            'model': model.state_dict(),
            'decoder': model.decoder.state_dict(),
            'encoder': model.encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(data_pack, f'runs/model-{c_epoch}.pt')
        attar_print(
            f'Done Saving model continue [*] \n Local : runs/model-{c_epoch}.pt', color=Cp.GREEN)
    print('')


def train(epochs, batch_size, data_loader, device='cuda:0' if torch.cuda.is_available() else 'cpu',
          prp: int = None,
          cfg: str = 'cfg/AEA-B.yaml', lr: float = 1e-4, img_size: tuple = (1920, 1080)

          ):
    prp = prp if prp is not None else epochs // 50
    dtt = data_loader
    model = TModel(cfg=cfg, batch_size=batch_size, status=True).to(device)

    try:
        attar_print('Loading DataLoaderTorch Train ...')
        data_train = dtt.return_dataLoader_train()
        attar_print('Done Loading DataLoaderTorch Train [*]')
    except Warning:
        attar_print('Error While Loading DataLoaderTorch Train action : pass [!] ', color=Cp.RED)

    try:
        attar_print('Loading DataLoaderTorch Eval ...')
        data_eval = dtt.return_dataLoader_eval()
        attar_print('Done Loading DataLoaderTorch Eval [*]')
    except Warning:
        attar_print('Error While Loading DataLoaderTorch Eval action : pass [!] ', color=Cp.RED)

    iter_data_train = iter(data_train) if data_train is not None else None
    iter_data_eval = iter(data_eval) if data_eval is not None else None
    batch_loop_train = math.ceil(data_train.__len__() / batch_size) if data_train is not None else 0
    batch_loop_eval = math.ceil(data_eval.__len__() / batch_size) if data_eval is not None else 0
    loss_func = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr, weight_decay=1e-4)

    for epoch in range(epochs):
        forward(data_train=data_train, batch_size=batch_size, optimizer=optimizer, loss_func=loss_func,
                img_size=img_size, batch_loop_train=batch_loop_train, c_epoch=epoch, prp=prp, model=model,
                epochs=epochs,
                device=device)
