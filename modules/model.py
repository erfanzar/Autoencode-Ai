import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import pars_model, read_yaml, Cp,attar_print


class Encoder(nn.Module):
    def __init__(self, cfg, init_model_on_start: bool = True, start_channel: int = 3):
        super(Encoder, self).__init__()
        self.m = None
        self.cfg = cfg
        self.start_channel = start_channel
        if init_model_on_start:
            self.init_model()

    def init_model(self):
        print(f'{Cp.CYAN} Creating Encode Model...{Cp.RESET}')
        self.m = pars_model(cfg=self.cfg, detail='',
                            print_status=True, sc=self.start_channel)
        print(
            f'{Cp.GREEN} Creating Encode Model Done Successfully [*]{Cp.RESET}\n')

    def forward(self, x):
        # attar_print(f'Encoder',color=Cp.RED)
        for m in self.m:
            # attar_print(f'Runnig {m}\n')
            x = m(x)
            # attar_print(f'Runnig {type(m).__name__} , shape : {x.shape}')
        return x


class Decoder(nn.Module):
    def __init__(self, cfg, init_model_on_start: bool = True, start_channel: int = 256):
        super(Decoder, self).__init__()
        self.m = None
        self.start_channel = start_channel
        self.cfg = cfg
        if init_model_on_start:
            self.init_model()

    def init_model(self):
        print(f'{Cp.CYAN} Creating Decoder Model...{Cp.RESET}')
        self.m = pars_model(cfg=self.cfg, detail='',
                            print_status=True, sc=self.start_channel)
        print(
            f'{Cp.GREEN} Creating Decoder Model Done Successfully [*]{Cp.RESET}\n')

    def forward(self, x):
        # attar_print(f'Decoder',color=Cp.RED)
        for m in self.m:
            
            x = m(x)
            # attar_print(f'Runnig {type(m).__name__} , shape : {x.shape}')
        return x


class TModel(nn.Module):
    def __init__(self, cfg):
        super(TModel, self).__init__()
        cfg = read_yaml(cfg)

        self.decoder = Decoder(
            cfg=cfg['decode'], start_channel=cfg['end_channel'])
        self.encoder = Encoder(
            cfg=cfg['encode'], start_channel=cfg['start_channel'])

    def forward(self, x):
  
        x = self.decoder.forward(self.encoder.forward(x))
        return x

    def forward_encoder(self, x):
        x = self.encoder.forward(x)
        return x

    def forward_decoder(self, x):
        x = self.decoder.forward(x)
        return x
