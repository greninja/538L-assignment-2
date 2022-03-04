import itertools
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers

from functools import partial

from torch.utils.data import DataLoader
from jax.experimental import optimizers, stax

from dataloader import get_cifar10_datasets, get_mnist_datasets
import optax
from opacus.accountants import create_accountant
from opacus.data_loader import DPDataLoader

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]= "False"
rng = jax.random.PRNGKey(0)
    
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

def softmax_loss(model, params, batch):
    inputs, targets = batch
    logits = model.apply(params, None, inputs)
    # convert the outputs to one hot shape according to the same shape as
    # logits for vectorized dot product
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    logits = stax.logsoftmax(logits)  # log normalize
    return -jnp.mean(jnp.sum(logits * one_hot, axis=-1))  # cross entropy loss

def compute_accuracy(models, params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(model.apply(params, None, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

def clipped_grad(model, loss, params, l2_norm_clip, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(partial(loss, model))(params, single_example_batch)
    nonempty_grads, tree_def = tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm([jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = jnp.maximum(total_grad_norm / l2_norm_clip, 1.)
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_unflatten(tree_def, normalized_nonempty_grads)

def private_grad(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
    """Return differentially private gradients for params, evaluated on batch."""
    clipped_grads = vmap(partial(clipped_grad, model, loss), (None, None, 0))(params, l2_norm_clip,
                                                                              batch)
    clipped_grads_flat, grads_treedef = tree_flatten(clipped_grads)
    aggregated_clipped_grads = [g.sum(0) for g in clipped_grads_flat]
    rngs = random.split(rng, len(aggregated_clipped_grads))
    noised_aggregated_clipped_grads = [
        g + l2_norm_clip * noise_multiplier * random.normal(r, g.shape)
        for r, g in zip(rngs, aggregated_clipped_grads)
    ]
    normalized_noised_aggregated_clipped_grads = [
        g / batch_size for g in noised_aggregated_clipped_grads
    ]
    return tree_unflatten(grads_treedef, normalized_noised_aggregated_clipped_grads)

# updating the params of the model
def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    return opt_update(
        i,
        private_grad(
            model, 
            softmax_loss, 
            params, 
            batch, 
            rng, 
            l2_norm_clip, 
            noise_multiplier,
            batch_size), 
        opt_state
    )

def main():

    # load data
    train_dataset, test_dataset = get_mnist_datasets()

    # load train and test loaders
    train_loader = DPDataLoader(train_dataset, sample_rate=sampling_prob)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,
                             num_workers=4, pin_memory=False)

    x_a, y_b = next(iter(train_loader))

    # train batch to get init params
    train_batch = x_a.numpy()

    # load model
    model = hk.transform(mnist_model)
    init_params = model.init(rng, train_batch)
    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)

    # define loss
    loss = softmax_loss

    opt_state = opt_init(init_params)
    itercount = itertools.count()

    # train function
    private_update = jit(private_update)
    # train_fn = private_update
    # train_fn = jit(train_fn)

    # train loop
    for epoch in range(1, epochs + 1):
        for iteration in iter_per_epoch:
            batch = next(iter(train_loader))
            batch = batch.numpy()
            opt_state = private_update(
                key,
                next(itercount),
                opt_state,
                batch,
            )

main()


# NOT NEEDED since you can use Opacus's RDPAccountant.
# def compute_epsilon(epoch, num_train_eg, args):
#     """Computes epsilon value for given hyperparameters."""
#     steps = epoch * num_train_eg // args.batch_size
#     if args.noise_multiplier == 0.0:
#         return float('inf')
#     orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
#     sampling_probability = args.batch_size / num_train_eg
#     rdp = compute_rdp(q=sampling_probability,
#                       noise_multiplier=args.noise_multiplier,
#                       steps=steps,
#                       orders=orders)
#     # Delta is set to approximate 1 / (number of training points).
#     return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]

def train():

def test():

def accountant():