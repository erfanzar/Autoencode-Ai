import imp
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from modules.commons import *
import numba as nb
import yaml

Any = [list, dict, int, float, str]


class Cp:
    Type = 1
    BLACK = f'\033[{Type};30m'
    RED = f'\033[{Type};31m'
    GREEN = f'\033[{Type};32m'
    YELLOW = f'\033[{Type};33m'
    BLUE = f'\033[{Type};34m'
    MAGENTA = f'\033[{Type};35m'
    CYAN = f'\033[{Type};36m'
    WHITE = f'\033[{Type};1m'
    RESET = f"\033[{Type};39m"


def read_yaml(path: str):
    with open(path, 'r') as r:
        data = yaml.full_load(r)
    return data

def attar_print(*args , color=Cp.GREEN,end_color=Cp.RESET,end = '\n'):
    print(*(f'{color if i == 0 else ""}{arg}{end_color}' for i,arg in enumerate(args)),end=end)

def print_model(model, args, form, times, index):
    print('{}  {:<5}{:<10}{:<10}{:>10}    -    {:<25} \n'.format(f'\033[1;39m', f"{index}", f"{form}", f"{times}",
                                                                f"{Cp.BLUE}{model}{Cp.RESET}",
                                                                f"{args}"))


def arg_creator(arg: list = None, prefix=None):
    # print(*((f' {v},' if i != len(arg) else f'{v}') for i, v in enumerate(arg)))
    created_args = f''.join(
        (((f'{prefix if prefix is not None else ""},{v},' if i == 0 else f'{v},') if i != len(arg) - 1 else f'{v}') for
         i, v in enumerate(arg)))
    return created_args


def pars_model(cfg, detail: str = None, print_status: bool = False, sc: int = 3):
    saves = []
    model = nn.ModuleList()
    c_req = ['Conv', 'TConv']
    if detail is not None:
        print(detail, end='')
    for i, c in enumerate(cfg):
        f, t, m, arg = c
        if print_status: print_model(m, arg, f, t, i)
        
        prefix = sc if m in c_req else ''
        arg_ = arg_creator(arg, prefix=prefix)
        model_name = f'{m}({arg_})'
        if not print_status :print(f"Adding : {model_name}")
      
        sc = arg[0] if m in c_req else sc
     
        m = eval(model_name)
        model.append(m)
    return model