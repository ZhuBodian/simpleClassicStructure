from torchvision import transforms
import torchvision
from torch.utils import data
import time
import numpy as np
import torch
import sys
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt

from IPython import display


def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=4))


class Accumulator:
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """Record multiple running times."""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def accuracy(y_hat, y):
    """Compute the number of correct predictions.
    y是标签向量，y_hat是矩阵，y_hat某一行的真正标签，为该行最大值所对应的列数，仅计算准确个数，不计算“准确率”
    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype))) / y_hat.shape[0]


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')



def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


class MyTimer:
    def __init__(self, timer_reason):
        self._start_timer = time.time()
        self._stop_timer = time.time()
        self.timer_reason = timer_reason

    def stop(self):
        self._stop_timer = time.time()
        print(f'"{self.timer_reason}" cost time: {self._stop_timer - self._start_timer}s')


class ProcessBar:
    def __init__(self, n, delta_ratio=None):
        self.n = n
        if delta_ratio is None:
            self.delta = int(self.n * 0.25) + 1
        else:
            assert 0 < delta_ratio < 1, 'delta_ratio应该在0与1之间'
            self.delta = int(self.n * delta_ratio) + 1

        idx = self.delta
        self.diplay_points = []
        for i in range(int(n/self.delta) + 1):
            self.diplay_points.append(idx)
            idx += self.delta

        self.displat_idx = 0

    def display(self, i, **kwargs):
        if i == self.diplay_points[self.displat_idx]:
            block_length = 50

            progress = (i / self.n) * 100
            temp = int(i / self.n * block_length)
            finish = "▓" * temp
            need_do = "-" * (block_length - temp)
            print(f'进度：{progress:^3.0f}%[{finish}->{need_do}] {i}/{self.n}；', end='')
            for key, value in kwargs.items():
                print(f'{key}：{value:.5f}，', end='')

            print('\n', end='')

            self.displat_idx += 1

# 就只是改变了一下调用方式，这种更符合平常习惯，即argmax(x)，用起来比x.argmax()顺手
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
