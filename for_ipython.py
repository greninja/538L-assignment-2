import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers

from functools import partial

from torch.utils.data import DataLoader

from jax.experimental import optimizers, stax

from dataloader import get_cifar10_datasets
import optax
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]= "False"
rng = jax.random.PRNGKey(0)

batch_size = 128
L = 128 # lot size
N = 50000 # total dataset size (for CIFAR10f)
sampling_prob = L / N 

# accountant creation
accountant = create_accountant("rdp")
std_dev = 1.0
max_grad_norm = 1.0 # is equal to C in DPSGD paper
noise_multiplier = std_dev / max_grad_norm

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

def mnist_model(input_features):
    return hk.Sequential([
        hk.Conv2D(16, (8, 8), padding='SAME', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Conv2D(32, (4, 4), padding='VALID', stride=(2, 2)),
        jax.nn.relu,
        hk.MaxPool(2, 1, padding='VALID'),  # matches stax
        hk.Flatten(),
        hk.Linear(32),
        jax.nn.relu,
        hk.Linear(10),
    ])(input_features)

# load data
train_dataset, test_dataset = get_cifar10_datasets()

# load train and test loaders
train_loader = DPDataLoader(train_dataset, sample_rate=sampling_prob)
# test_loader = DPDataLoader(test_dataset, sample_rate=sampling_prob)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                         num_workers=4, pin_memory=False)

x_a, y_b = next(iter(train_loader))

# for i, batch in enumerate(train_loader): # batch instead of (inputs, targets) for loss function
#     break

# batch = train_dataset.data[:10].astype(np.int32)
train_batch = x_a.numpy()

# load model
model = hk.transform(cifar10_model)
init_params = model.init(rng, train_batch)



