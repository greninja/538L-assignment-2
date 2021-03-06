from torchvision import transforms, datasets
from typing import *
import torch
import os
import glob
import numpy as np 
from torch.utils.data import Dataset
from PIL import Image

# CIFAR-10
_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]

CIFAR_TRAIN_TRANSFORM = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        ])
CIFAR_TEST_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)
        ])

# MNIST
_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]

MNIST_TRAIN_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ])
MNIST_TEST_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ])

def get_cifar10_datasets():    
    train_dataset = datasets.CIFAR10("./data", train=True, download=False, transform=CIFAR_TRAIN_TRANSFORM)
    test_dataset = datasets.CIFAR10("./data", train=False, download=False, transform=CIFAR_TEST_TRANSFORM)
    return train_dataset, test_dataset

def get_mnist_datasets():    
    train_dataset = datasets.MNIST("./data", train=True, download=False, transform=MNIST_TRAIN_TRANSFORM)
    test_dataset = datasets.MNIST("./data", train=False, download=False, transform=MNIST_TEST_TRANSFORM)
    return train_dataset, test_dataset    