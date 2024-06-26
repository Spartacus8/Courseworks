import matplotlib.pyplot as plt
import numpy as np

import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision.transforms import ToTensor

def load_mnist(transform):
    trainset = MNIST(root='./data', train=True,
                       download=True, transform=transform)
    testset = MNIST(root='./data', train=False,
                      download=True, transform=transform)
    return trainset, testset


def load_cifar10(transform):
    trainset = CIFAR10(root='./data', train=True,
                       download=True, transform=transform)
    testset = CIFAR10(root='./data', train=False,
                      download=True, transform=transform)
    return trainset, testset


def load_fashion_mnist(transform):
    trainset = FashionMNIST(root='./data', train=True,
                       download=True, transform=transform)
    testset = FashionMNIST(root='./data', train=False,
                      download=True, transform=transform)
    return trainset, testset


# functions to show an image
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
