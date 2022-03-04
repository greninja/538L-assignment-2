# def initialize_mlp(sizes, key):
#     """ Initialize the weights of all layers of a linear layer network """
#     keys = random.split(key, len(sizes))
#     # Initialize a single layer with Gaussian weights -  helper function
#     def initialize_layer(m, n, key, scale=1e-2):
#         w_key, b_key = random.split(key)
#         return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
#     return [initialize_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

# layer_sizes = [784, 512, 512, 10]
# # Return a list of tuples of layer weights
# params = initialize_mlp(layer_sizes, key)

# def forward_pass(params, in_array):
#     """ Compute the forward pass for each example individually """
#     activations = in_array

#     # Loop over the ReLU hidden layers
#     for w, b in params[:-1]:
#         activations = relu_layer([w, b], activations)

#     # Perform final trafo to logits
#     final_w, final_b = params[-1]
#     logits = np.dot(final_w, activations) + final_b
#     return logits - logsumexp(logits)

# # Make a batched version of the `predict` function
# batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)
