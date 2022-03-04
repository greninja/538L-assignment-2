import time
import itertools
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

from torch.nn.utils import clip_grad_norm_
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader

from torch.utils.data import DataLoader

# HYPERPARAMS
# Generate key which is used to generate random numbers
num_classes = 10

batch_size = 128 # also lot size
N = 60000 # total dataset size (for MNIST)
sampling_prob = batch_size / N
epochs = 1
iter_per_epoch = int(N / batch_size)  
learning_rate = 0.01

# accountant creation
accountant = create_accountant("rdp")
std_dev = 1.0
max_grad_norm = 1.0 # is equal to C in DPSGD paper
noise_multiplier = std_dev / max_grad_norm

# MNIST
_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]

MNIST_TRAIN_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ])

criterion = nn.CrossEntropyLoss()

class MNIST_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

model = MNIST_NET()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_dataset = datasets.MNIST("./data", train=True, download=False, transform=MNIST_TRAIN_TRANSFORM)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

for e in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        for param in model.parameters():
            param.accumulated_grads = []
        
        inputs, targets = batch

        # Run the microbatches
        for (x,y) in zip(inputs, targets):
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
            y = torch.unsqueeze(y, dim=0)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
        
            # Clip each parameter's per-sample gradient
            for param in model.parameters():
                per_sample_grad = param.grad.detach().clone()
                clip_grad_norm_(per_sample_grad, max_norm=max_grad_norm)  # in-place
                param.accumulated_grads.append(per_sample_grad)
            
        # Aggregate back
        for param in model.parameters():
            param.grad = torch.stack(param.accumulated_grads, dim=0)

        # Now we are ready to update and add noise!
        for param in model.parameters():
            param = param - learning_rate * param.grad
            param += torch.normal(mean=0, std=noise_multiplier * max_grad_norm)
            
            param.grad = 0  # Reset for next iteration
        
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampling_prob)