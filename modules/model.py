import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import pars_model, read_yaml


class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.m = None
        self.cfg = cfg
        self.init_model()

    def init_model(self):
        self.m = pars_model(cfg=self.cfg, detail='', print_status=True)

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.m = None
        self.cfg = cfg
        self.init_model()

    def init_model(self):
        self.m = pars_model(cfg=self.cfg, detail='', print_status=True)

    def forward(self, x):
        pass


class TModel(nn.Module):
    def __init__(self, cfg):
        super(TModel, self).__init__()
        cfg = read_yaml(cfg)
        self.decoder = Decoder(cfg=cfg['decode'])
        self.encoder = Encoder(cfg=cfg['encode'])

    def forward(self, x):
        pass
