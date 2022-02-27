import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers

from functools import partial

from torch.utils.data import DataLoader

from dataloader import get_cifar10_datasets

rng = jax.random.PRNGKey(0)
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]= "False"

def cifar10_model(input_features):
    return hk.Sequential([
        hk.Conv2D(64, (5, 5), padding='SAME', stride=(1, 1)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),
        hk.Conv2D(64, (5, 5), padding='SAME', stride=(1, 1)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),
        hk.Flatten(),
        hk.Linear(384),
        jax.nn.relu,
        hk.Linear(384)
    ])(input_features)

# def main(args):

batch_size = 128

# load data
train_dataset, test_dataset = get_cifar10_datasets()

# load train and test loaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,
                          num_workers=4, pin_memory=False)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                         num_workers=4, pin_memory=False)

for i, batch in enumerate(train_loader): # batch instead of (inputs, targets) for loss function
    break

# batch = train_dataset.data[:10].astype(np.int32)
train_batch = batch[0].numpy()

# load model
model = hk.transform(cifar10_model)
init_params = model.init(rng, train_batch)

# define loss
loss = softmax_loss




def softmax_loss(model, params, batch):
    inputs, targets = batch

def accuracy():

def train():

def test():    


def clipped_grad():

def noisy_grad():

def accountant():
