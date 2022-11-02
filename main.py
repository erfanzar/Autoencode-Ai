from utils.dataset import DataLoaderTorch
from modules.model import TModel, Encoder, Decoder

if __name__ == "__main__":
    print('Somethings...')
    dtt = DataLoaderTorch(dataset="MNIST")
    model = TModel(cfg='cfg/AEA-S.yaml')
