import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, random, vmap
from jax.example_libraries import optimizers

from functools import partial

from torch.utils.data import DataLoader

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

def noisy_grad(model, loss, params, batch, rng, l2_norm_clip, noise_multiplier, batch_size):
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
def update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    return opt_update(
        i,
        grad_fn(model, loss, params, batch, rng, args.l2_norm_clip, args.noise_multiplier,
                args.batch_size), opt_state)

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

def main(args):

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
    
    # load optimizer
    optimizer = optax.sgd(learning_rate)
    params = {'w': jnp.ones((init_params,))}
    opt_state = optimizer.init(params)

    # define loss
    loss = softmax_loss

    # train loop


