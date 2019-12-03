from torch import utils
from torchvision import datasets, transforms

def read_dataset(trans, path, size):
    train_data = datasets.MNIST(path, train=True, download=True, transform=trans)
    test_data = datasets.MNIST(path, train=False, download=True, transform=trans)

    loader = utils.data.DataLoader(train_data, batch_size=size)

    return next(iter(loader))
