import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import pars_model, read_yaml, Cp, attar_print


class Encoder(nn.Module):
    def __init__(self, cfg, init_model_on_start: bool = True, start_channel: int = 3, is_linear: bool = False,
                 status: bool = False):
        super(Encoder, self).__init__()
        self.m = None
        self.cfg = cfg
        self.start_channel = start_channel
        self.status = status
        self.is_linear = is_linear
        if init_model_on_start:
            self.init_model()

    def init_model(self):
        print(f'{Cp.CYAN} Creating Encode Model...{Cp.RESET}')
        self.m = pars_model(cfg=self.cfg, detail='',
                            print_status=self.status, sc=self.start_channel)
        print(
            f'{Cp.GREEN} Creating Encode Model Done Successfully [*]{Cp.RESET}\n')

    def forward(self, x):
        # attar_print(f'Encoder',color=Cp.RED)
        if self.is_linear: x = x.view(x.shape[0], -1)
        for m in self.m:
            # attar_print(f'before Run {type(m).__name__} , shape : {x.shape}')
            x = m(x)
            # attar_print(f'after Run {type(m).__name__} , shape : {x.shape}')
            # attar_print(' - ' * 20)
        return x


class Decoder(nn.Module):
    def __init__(self, cfg, init_model_on_start: bool = True, start_channel: int = 256, is_linear: bool = False,
                 status: bool = False):
        super(Decoder, self).__init__()
        self.is_linear = is_linear
        self.m = None
        self.start_channel = start_channel
        self.status = status
        self.cfg = cfg
        if init_model_on_start:
            self.init_model()

    def init_model(self):
        print(f'{Cp.CYAN} Creating Decoder Model...{Cp.RESET}')
        self.m = pars_model(cfg=self.cfg, detail='',
                            print_status=self.status, sc=self.start_channel)
        print(
            f'{Cp.GREEN} Creating Decoder Model Done Successfully [*]{Cp.RESET}\n')

    def forward(self, x):
        # attar_print(f'Decoder',color=Cp.RED)
        if self.is_linear: x = x.view(x.shape[0], -1)
        for m in self.m:
            # attar_print(f'before Run {type(m).__name__} , shape : {x.shape}')
            x = m(x)
            # attar_print(f'after Run {type(m).__name__} , shape : {x.shape}')
            # attar_print(' - '*20)
        return x


class TModel(nn.Module):
    def __init__(self, cfg, batch_size: int = 64, status: bool = False):
        super(TModel, self).__init__()
        cfg = read_yaml(cfg)
        self.batch_size = batch_size

        self.encoder = Encoder(
            cfg=cfg['encode'], start_channel=cfg['start_channel'], is_linear=cfg['is_linear'], status=status)
        self.decoder = Decoder(
            cfg=cfg['decode'], start_channel=cfg['end_channel'], is_linear=cfg['is_linear'], status=status)

    def forward(self, x):
        x = self.decoder.forward(self.encoder.forward(x))
        return x

    def forward_encoder(self, x):
        x = self.encoder.forward(x)
        return x

    def forward_decoder(self, x):
        x = self.decoder.forward(x)
        return x
