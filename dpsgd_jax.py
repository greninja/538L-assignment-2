import jax
import time
import itertools
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random

from jax.tree_util import Partial 

# Import some additional JAX and dataloader helpers
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax)

# for privacy accounting
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader

from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten

import torch
from torchvision import datasets, transforms

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

def softmax_loss(model, params, batch):
    inputs, targets = batch
    logits = model(params, inputs)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    logits = stax.logsoftmax(logits)  # log normalize
    return -jnp.mean(jnp.sum(logits * one_hot, axis=-1))  # cross entropy loss

def gradient_clipping(model, loss_fn, params, l2_norm_bound, per_example_batch):
    """Compute gradient for a single example and clip its norm to 'l2_norm_bound' 
    (Note the single example batch is automatically generated via vmap function"""

    reshaped_input = jnp.expand_dims(per_example_batch[0], axis=1)
    target = per_example_batch[1]
    reshaped_batch = (reshaped_input, target)
    grads = grad(loss_fn, argnums=1)(model, params, reshaped_batch)

    grad_sq_sum = 0
    for i in grads:
        for j in i:
            grad_sq_sum += jnp.sum(jnp.square(j.ravel()))
    total_grad_norm = jnp.sqrt(grad_sq_sum)

    scale_factor = jnp.maximum(total_grad_norm / l2_norm_bound, 1.)

    clipped_grad = []
    for i in grads:
        u = []
        if len(i) > 0:
            for j in i:
                j/=scale_factor
                u.append(j)
        clipped_grad.append(u)

    return clipped_grad

def compute_private_grad(model, loss_fn, params, batch, key, l2_norm_bound, noise_multiplier, batch_size):
    """Return differentially private gradients for params, evaluated on batch."""

    # vmap function
    clipped_grads = vmap(gradient_clipping,
                        in_axes=(None, None, None, None, 0),
                        out_axes=0)(model, loss_fn, params, l2_norm_bound, batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    keys = random.split(key, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [g + l2_norm_bound * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(keys, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

# HYPERPARAMS
# Generate key which is used to generate random numbers
key = random.PRNGKey(42)
num_classes = 10

batch_size = 128 # also lot size
N = 60000 # total dataset size (for MNIST)
delta = 1/N
sampling_prob = batch_size / N
num_epochs = 1
iter_per_epoch = int(N / batch_size)
learning_rate = 0.01

# accountant creation
accountant = create_accountant("rdp")
std_dev = 1.0
l2_norm_bound = 1.0 # is equal to C in DPSGD paper
noise_multiplier = std_dev / l2_norm_bound
step_size = 1e-3

# MNIST
_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]

MNIST_TRAIN_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ])

train_dataset = datasets.MNIST("./data", train=True, download=False, transform=MNIST_TRAIN_TRANSFORM)
train_loader = DPDataLoader(train_dataset, sample_rate=sampling_prob)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
                   ])),
    batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
                   ])),
    batch_size=batch_size, shuffle=True, drop_last=True)

# conv_net model
init_fun, model = stax.serial(Conv(32, (5, 5), (2, 2), padding="SAME"),
                              BatchNorm(), Relu,
                              Conv(32, (5, 5), (2, 2), padding="SAME"),
                              BatchNorm(), Relu,
                              Conv(10, (3, 3), (2, 2), padding="SAME"),
                              BatchNorm(), Relu,
                              Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
                              Flatten,
                              Dense(num_classes),
                              LogSoftmax)

_, params = init_fun(key, (1, 1, 28, 28))

opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

""" Implements a learning loop over epochs. """
# Initialize placeholder for loggin
log_acc_train, log_acc_test, train_loss = [], [], []

# Get the initial set of parameters
params = get_params(opt_state)
itercount = itertools.count()

eps_arr = []

# Main loop
for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        i = next(itercount) # tracks the current step
        key = random.fold_in(key, i)  # get new key for new random numbers

        inputs = jnp.array(inputs)
        targets = jnp.array(targets)
        batch = (inputs, targets)

        # setting the loss function
        clipped_noisy_grads = jit(compute_private_grad(model, 
                                                       softmax_loss, 
                                                       params, 
                                                       batch,
                                                       key, 
                                                       l2_norm_bound, 
                                                       noise_multiplier, 
                                                       batch_size))
        opt_state = opt_update(i, clipped_noisy_grads, opt_state)
        params = get_params(opt_state)

        # take a step for acountant and measure privacy spent
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sampling_prob)
        eps_till_now, best_alpha = accountant.get_privacy_spent(delta=delta)
        eps_arr.append(eps_till_now)


eps_arr = main(num_epochs, opt_state)
x = np.arange(num_epochs)
plt.plot(x, eps_arr)
plt.savefig("eps vs epochs")
print("Epoch {} | T: {:0.2f} | eps till now: {:0.2f}".format(
                                                            epoch+1,
                                                            epoch_time,
                                                            eps_till_now))

# LIST OF PLOTS:
# - (unclipped) gradient plots
# - eps vs iterations/steps
# - alpha order RDP