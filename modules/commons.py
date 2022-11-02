import torch
import torch.nn as nn


# import numpy as np
# import numba as nb


class Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int = None, g: int = 1, act: torch.nn = None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, (0 if not k == 3 else 1) if p is None else p, groups=g)
        self.act = eval(f"nn.{act}" if act is not None else 'nn.Identity()')

    def forward(self, x):
        return self.act(self.conv(x))


class TConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int, p: int = None, op: int = None, g: int = 1,
                 act: torch.nn = None):
        super(TConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=k, stride=s,
                                       padding=(0 if not k == 3 else 1) if p is None else p,
                                       output_padding=(0 if not k == 3 else 1) if op is None else op, groups=g)
        self.act = eval(f"nn.{act}" if act is not None else 'nn.Identity()')

    def forward(self, x):
        return self.act(self.conv(x))


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return self.s(x)


class CaConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int, p: int = None, g: int = 1,
                 e: float = 0.5, act: torch.nn = None):
        super(CaConv, self).__init__()
        ce = int(c2 * e)
        self.c1 = Conv(c1=c1, c2=ce, k=k, s=1, act=act, g=1)
        self.c2 = Conv(c1=c1, c2=ce, k=k, s=1, act=act, g=1)
        self.m = Conv(c1=ce, c2=c2, k=k, s=s, act=act, p=p, g=g)

    def forward(self, x):
        out = self.m(torch.cat((self.c1(x), self.c2(x)), 1))
        return out
