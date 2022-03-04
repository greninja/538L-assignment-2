import time
import itertools
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad, random

# Import some additional JAX and dataloader helpers
from jax.scipy.special import logsumexp
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, Dense, Flatten,
                                   Relu, LogSoftmax)

from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader

from functools import partial
import torch
from torchvision import datasets, transforms

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# HYPERPARAMS 
# Generate key which is used to generate random numbers
key = random.PRNGKey(42)
num_classes = 10

batch_size = 128 # also lot size
N = 60000 # total dataset size (for MNIST)
sampling_prob = batch_size / N
epochs = 100
iter_per_epoch = int(N / batch_size)  
learning_rate = 0.01

# accountant creation
accountant = create_accountant("rdp")
std_dev = 1.0
l2_norm_clip = 1.0 # is equal to C in DPSGD paper
noise_multiplier = std_dev / l2_norm_clip

def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def relu_layer(params, x):
    """ Simple ReLu layer for single sample """
    return ReLU(jnp.dot(params[0], x) + params[1])

# def forward_pass(params, images):
#     return model(params, images)

# @jit
# def update(params, x, y, opt_state):
#     """ Compute the gradient for a batch and update the parameters """
#     value, grads = value_and_grad(loss)(params, x, y)
#     opt_state = opt_update(0, grads, opt_state)
#     return get_params(opt_state), opt_state, value

# def accuracy(params, data_loader):
#     """ Compute the accuracy for the CNN case (no flattening of input)"""
#     acc_total = 0
#     for batch_idx, (data, target) in enumerate(data_loader):
#         images = jnp.array(data)
#         targets = one_hot(jnp.array(target), num_classes)

#         target_class = jnp.argmax(targets, axis=1)
#         predicted_class = jnp.argmax(batch_forward(params, images), axis=1)
#         acc_total += jnp.sum(predicted_class == target_class)
#     return acc_total/len(data_loader.dataset)

# def loss(params, images, targets):
#     preds = model(params, images)
#     return -jnp.sum(preds * targets)

def softmax_loss(model, params, batch):
    inputs, targets = batch
    print(inputs.shape)
    logits = model(params, inputs)
    # convert the outputs to one hot shape according to the same shape as
    # logits for vectorized dot product
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    logits = stax.logsoftmax(logits)  # log normalize
    return -jnp.mean(jnp.sum(logits * one_hot, axis=-1))  # cross entropy loss

def clipped_grad(model, loss, params, l2_norm_clip, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(partial(loss, model))(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)

def private_grad(model, loss, params, batch, key, l2_norm_clip, noise_multiplier, batch_size):

    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = vmap(partial(clipped_grad, model, loss), (None, None, 0))(params, l2_norm_clip,
                                                                              batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    keys = random.split(key, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(keys, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

@jit
# updating the params of the model
def private_update(key, i, opt_state, batch):
    params = get_params(opt_state)
    key = random.fold_in(key, i)  # get new key for new random numbers
    return opt_update(
        i,
        private_grad(
            model,
            softmax_loss, 
            params, 
            batch, 
            key,
            l2_norm_clip, 
            noise_multiplier,
            batch_size), 
        opt_state
    )

def main(num_epochs, opt_state):
    """ Implements a learning loop over epochs. """
    # Initialize placeholder for loggin
    log_acc_train, log_acc_test, train_loss = [], [], []

    # Get the initial set of parameters
    params = get_params(opt_state)

    # # Get initial accuracy after random init
    # train_acc = accuracy(params, train_loader)
    # test_acc = accuracy(params, test_loader)
    # log_acc_train.append(train_acc)
    # log_acc_test.append(test_acc)

    itercount = itertools.count()

    # Loop over the training epochs
    for epoch in range(num_epochs):
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            i = next(itercount) # tracks the current step
            x = jnp.array(data)
            y = one_hot(jnp.array(target), num_classes)
            batch = (x, y)
            # params, opt_state, loss = private_update(key, i, opt_state, params, x, y)
            opt_state = private_update(key, i, opt_state, batch)

        # epoch_time = time.time() - start_time
        # train_acc = accuracy(params, train_loader)
        # test_acc = accuracy(params, test_loader)
        # log_acc_train.append(train_acc)
        # log_acc_test.append(test_acc)
        # print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(epoch+1, epoch_time,
        #                                                             train_acc, test_acc))
        print("Epoch {} | T: {:0.2f} | Train A: {:0.3f} | Test A: {:0.3f}".format(epoch+1, epoch_time,
                                                                    train_acc, test_acc))

    return train_loss, log_acc_train, log_acc_test

# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('./data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ])),
#     batch_size=batch_size, shuffle=True)

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

# init_fun, model = stax.serial(Conv(32, (5, 5), (2, 2), padding="SAME"),
#                               BatchNorm(), Relu,
#                               Conv(32, (5, 5), (2, 2), padding="SAME"),
#                               BatchNorm(), Relu,
#                               Conv(10, (3, 3), (2, 2), padding="SAME"),
#                               BatchNorm(), Relu,
#                               Conv(10, (3, 3), (2, 2), padding="SAME"), Relu,
#                               Flatten,
#                               Dense(num_classes),
#                               LogSoftmax)
# _, params = init_fun(key, (1, 28, 28))





def initialize_mlp(sizes, key):
    """ Initialize the weights of all layers of a linear layer network """
    keys = random.split(key, len(sizes))
    # Initialize a single layer with Gaussian weights -  helper function
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
# Return a list of tuples of layer weights
params = initialize_mlp(layer_sizes, key)

def forward_pass(params, in_array):
    """ Compute the forward pass for each example individually """
    activations = in_array

    # Loop over the ReLU hidden layers
    for w, b in params[:-1]:
        activations = relu_layer([w, b], activations)

    # Perform final trafo to logits
    final_w, final_b = params[-1]
    logits = np.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

# Make a batched version of the `predict` function
batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)




step_size = 1e-3
opt_init, opt_update, get_params = optimizers.adam(step_size)
opt_state = opt_init(params)
num_epochs = 1

""" Implements a learning loop over epochs. """
# Initialize placeholder for loggin
log_acc_train, log_acc_test, train_loss = [], [], []

# Get the initial set of parameters
params = get_params(opt_state)
itercount = itertools.count()

# Loop over the training epochs
for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        i = next(itercount) # tracks the current step
        x = jnp.array(data)
        y = one_hot(jnp.array(target), num_classes)
        batch = (x, y)
        # params, opt_state, loss = private_update(key, i, opt_state, params, x, y)
        opt_state = private_update(key, i, opt_state, batch)


