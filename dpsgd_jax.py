import jax
import time
import itertools
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random

# from jax.tree_util import Partial
from functools import partial 

# Import some additional JAX and dataloader helpers
from jax.scipy.special import logsumexp
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import (BatchNorm, Conv, Dense, Flatten, Relu, LogSoftmax)

# for privacy accounting
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader
from opacus.accountants.analysis import rdp as privacy_analysis

from jax.tree_util import tree_flatten, tree_multimap, tree_unflatten

import torch
from torchvision import datasets, transforms
from typing import List, Tuple, Union

import matplotlib.pyplot as plt 
import seaborn as sns
# sns.set()
sns.set_theme(style="whitegrid", palette="pastel")

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

def get_epsilon_spent_for_every_alpha(orders: Union[List[float], float], 
                                      rdp: Union[List[float], float], 
                                      delta: float) -> Tuple[List[float], float]:

    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    eps_arr = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )
    idx_opt = np.nanargmin(eps_arr)  # Ignore NaNs
    # return eps[idx_opt], orders_vec[idx_opt]
    return eps_arr, orders_vec[idx_opt]

def compute_test_accuracy(model, params, test_loader):
    correct = 0
    total = 0
    for (inputs, targets) in test_loader:
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)  
        predicted_class = jnp.argmax(model(params, inputs), axis=1)
    
        correct += jnp.sum(predicted_class == targets)
        total += len(targets)
    return (correct/total) * 100

def softmax_loss(model, params, batch):
    inputs, targets = batch
    logits = model(params, inputs)
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    logits = stax.logsoftmax(logits)  # log normalize
    return -jnp.mean(jnp.sum(logits * one_hot, axis=-1))  # cross entropy loss

def gradient_clipping(model, loss_fn, params, l2_norm_bound, per_example_batch):
    """Compute gradient for a single example and clip its norm to 'l2_norm_bound' 
    (Note the single example batch is automatically generated via vmap function)"""

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
        u = tuple()
        if len(i) > 0:
            for j in i:
                j/=scale_factor
                u += (j,)
        clipped_grad.append(u)

    return clipped_grad, total_grad_norm

def compute_private_gradients(model, loss_fn, params, batch, key, l2_norm_bound, noise_multiplier, batch_size):
    """Return differentially private gradients for params, evaluated on batch."""

    # vmap function
    clipped_grads, batch_grad_norms = vmap(gradient_clipping,
                         in_axes=(None, None, None, None, 0),
                         out_axes=0)(model, loss_fn, params, l2_norm_bound, batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    summed_grads = [g.sum(0) for g in clipped_grads_flat]
    
    # get keys for each draw of random noise
    keys = random.split(key, len(summed_grads))
    
    noised_summed_clipped_grads = [g + (l2_norm_bound * noise_multiplier * random.normal(r, g.shape))
                                         for r, g in zip(keys, summed_grads)]
    averaged_noisy_grads = [g / batch_size for g in noised_summed_clipped_grads]
    averaged_noisy_clipped_grads = tree_unflatten(grads_treedef, averaged_noisy_grads)
    return averaged_noisy_clipped_grads, batch_grad_norms

def dpsgd_update_step(key, iteration, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(key, i)
    averaged_noisy_clipped_grads, batch_grad_norms = compute_private_gradients(model, 
                                                                               softmax_loss,
                                                                               params,
                                                                               batch, 
                                                                               rng,
                                                                               l2_norm_bound,
                                                                               noise_multiplier,
                                                                               batch_size)
    next_opt_state = opt_update(i, averaged_noisy_clipped_grads, opt_state)
    return next_opt_state, batch_grad_norms

# HYPERPARAMS
# Generate key which is used to generate random numbers
key = random.PRNGKey(42)
num_classes = 10

batch_size = 128 # also lot size
N = 60000 # total dataset size (for MNIST)
delta = 1/N
sample_rate = batch_size / N
num_epochs = 5
iter_per_epoch = int(N / batch_size)
learning_rate = 0.01

# accountant creation
accountant = create_accountant("rdp")
std_dev = 1.0
l2_norm_bound = 1.0 # is equal to C in DPSGD paper
noise_multiplier = std_dev / l2_norm_bound
step_size = 1e-3

# alphas for RDP
alphas = range(2,33)

# MNIST
_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]

MNIST_TRAIN_TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)
        ])

train_dataset = datasets.MNIST("./data", train=True, download=False, transform=MNIST_TRAIN_TRANSFORM)
train_loader = DPDataLoader(train_dataset, sample_rate=sample_rate)

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
    batch_size=100, shuffle=True, drop_last=True)

# Conv Net model
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

# batch dim is set to 1 since we'll be passing an example at a time to get per-example gradients 
_, params = init_fun(key, (1, 1, 28, 28))

opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)

# Get the initial set of parameters
params = get_params(opt_state)
itercount = itertools.count()

# arrays for storing data required for plotting
eps_arr = []
epoch_avg_gn_arr = []
test_acc_arr = []

jitted_update_step = jit(dpsgd_update_step)

rdp = np.zeros_like(alphas, dtype=float)

# Main loop
for epoch in range(1, num_epochs+1):
    start_time = time.time()
    epoch_average_gn = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        i = next(itercount) # tracks the current step

        inputs = jnp.array(inputs)
        targets = jnp.array(targets)
        batch = (inputs, targets)

        # take a private gradient step and get the updated optimizer state and batch gradient norms
        opt_state, batch_grad_norms = jitted_update_step(key, i, opt_state, batch)
        epoch_average_gn += jnp.sum(batch_grad_norms)

        # get updated model params
        params = get_params(opt_state)

        # # take a step for acountant and measure privacy spent
        # accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        # eps_till_now, best_alpha = accountant.get_privacy_spent(delta=delta)
        # eps_arr.append(eps_till_now)
    
        rdp += privacy_analysis.compute_rdp(q=sample_rate, 
                                            noise_multiplier=noise_multiplier,
                                            steps=i,
                                            orders=alphas)

        eps_arr, best_alpha = get_epsilon_spent_for_every_alpha(orders=alphas,
                                                                rdp=rdp,
                                                                delta=delta)

        print("eps till now {:0.5f}".format(eps_till_now))

    epoch_average_gn /= N
    epoch_avg_gn_arr.append(epoch_average_gn)
    epoch_test_accuracy = compute_test_accuracy(model, params, test_loader)
    test_acc_arr.append(epoch_test_accuracy)

    print("Epoch {} | eps spent till now {:0.5f} | epoch grad norm {:0.3f} | test accuracy {:0.2f}".format(
           epoch, eps_till_now, epoch_average_gn, epoch_test_accuracy))    

# # PLOTTING CODE
# x = np.arange(num_epochs)
# plt.plot(x, eps_arr)
# plt.savefig("eps vs steps")

# LIST OF PLOTS:
# - (unclipped) gradient plots
# - eps vs iterations/steps
# - alpha order RDP