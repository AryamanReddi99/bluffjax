import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
import matplotlib.pyplot as plt
import seaborn as sns
import os

# sns.set_theme()

# training config
num_epochs = 10
num_minibatches = 4
num_seeds = 10

# dataset
x0 = jnp.zeros(10000)[:, None]
x1 = jnp.ones(10000)[:, None]
xboth = jnp.concatenate([x0, x1], axis=0)
x = jnp.tile(xboth, (2))


def encode(el):
    return (jnp.arange(2) == el).astype(jnp.float32)


# Single examples for visualization (one of each class)
# mlp uses binary encoding: [0,0] or [1,1] - shape (2,)
x_single_binary = jnp.array(
    [[0.0, 0.0], [1.0, 1.0]]
)  # Shape: (2, 2) - binary encoding for mlp
# mlp_binary uses one-hot encoding: [1,0] or [0,1] - shape (2,)
x_single_onehot = jnp.array(
    [[1.0, 0.0], [0.0, 1.0]]
)  # Shape: (2, 2) - one-hot encoding for mlp_binary


x_binary = jax.vmap(encode)(x)

# y = 1 - x
# y = x
# y0 = jnp.tile(jnp.array([0.75, 0.25]), (x0.shape[0], 1))
# y1 = jnp.tile(jnp.array([0.25, 0.75]), (x0.shape[0], 1))

# rand


# network
class MLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        activation = nn.relu

        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(
            self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        return x


# Helper function to extract hidden layer activations
def extract_hidden1(mlp, params, x):
    """Extract first hidden layer activations from MLP"""
    activation = nn.relu
    # params from TrainState has structure {'params': {...}}
    dense_params = params["params"]["Dense_0"]
    kernel = dense_params["kernel"]
    bias = dense_params["bias"]

    # Ensure x has batch dimension: (batch_size, input_dim)
    if x.ndim == 1:
        x = x[None, :]  # Add batch dimension: (1, input_dim)

    # Check kernel shape matches input dimension
    # kernel should be (input_dim, 64)
    # x is (batch_size, input_dim)
    # Result: (batch_size, 64)
    x = jnp.dot(x, kernel) + bias
    hidden1 = activation(x)
    return hidden1


def train(rng):
    # y
    rng, rng0, rng1 = jax.random.split(rng, 3)
    y0 = jax.random.uniform(rng0, 2)
    y1 = jax.random.uniform(rng1, 2)
    y = jnp.where(xboth == 0, y0, y1)

    # networks
    rng, rng_network_oh_init, rng_network_binary_init = jax.random.split(rng, 3)
    mlp = MLP(output_dim=y.shape[-1])
    mlp_params = mlp.init(rng_network_oh_init, x[0])
    tx = optax.adam(learning_rate=1e-3, eps=1e-5)
    train_state = TrainState.create(apply_fn=mlp.apply, params=mlp_params, tx=tx)

    mlp_binary = MLP(output_dim=y.shape[-1])
    mlp_binary_params = mlp_binary.init(rng_network_binary_init, x_binary[0])
    tx_binary = optax.adam(learning_rate=1e-3, eps=1e-5)
    train_state_binary = TrainState.create(
        apply_fn=mlp_binary.apply, params=mlp_binary_params, tx=tx_binary
    )

    def train_epoch(carry, epoch):
        train_state, train_state_binary, rng = carry
        rng, rng_shuffle = jax.random.split(rng)
        shuffle = jax.random.permutation(rng_shuffle, x.shape[0])
        shuffled_x = x[shuffle]
        shuffled_x_binary = x_binary[shuffle]
        shuffled_y = y[shuffle]

        x_minibatches = shuffled_x.reshape(num_minibatches, -1, x.shape[-1])
        x_binary_minibatches = shuffled_x_binary.reshape(
            num_minibatches, -1, shuffled_x_binary.shape[-1]
        )
        y_minibatches = shuffled_y.reshape(num_minibatches, -1, *y.shape[1:])

        def train_minibatch(carry, ins):
            (
                train_state,
                train_state_binary,
            ) = carry
            x_minibatch, x_binary_minibatch, y_minibatch = ins

            def loss(params, x, y):
                pred = mlp.apply(params, x)
                diff = pred - y
                return (diff**2).mean()

            def loss_binary(params, x, y):
                pred = mlp_binary.apply(params, x)
                diff = pred - y
                return (diff**2).mean()

            loss, loss_grad = jax.value_and_grad(loss)(
                train_state.params, x_minibatch, y_minibatch
            )
            loss_binary, loss_grad_binary = jax.value_and_grad(loss_binary)(
                train_state_binary.params, x_binary_minibatch, y_minibatch
            )

            train_state_update = train_state.apply_gradients(grads=loss_grad)
            train_state_binary_update = train_state_binary.apply_gradients(
                grads=loss_grad_binary
            )

            return (train_state_update, train_state_binary_update), (loss, loss_binary)

        (train_state, train_state_binary), (loss, loss_binary) = lax.scan(
            train_minibatch,
            (train_state, train_state_binary),
            (x_minibatches, x_binary_minibatches, y_minibatches),
        )

        # Capture activations at all epochs (we'll filter later)
        # Only capture single examples of each class for visualization
        # mlp uses binary encoding [0,0] or [1,1], mlp_binary uses one-hot [1,0] or [0,1]
        hidden1_onehot = extract_hidden1(
            mlp_binary, train_state_binary.params, x_single_onehot
        )
        hidden1_binary = extract_hidden1(mlp, train_state.params, x_single_binary)

        return (train_state, train_state_binary, rng), (
            loss.mean(),
            loss_binary.mean(),
            hidden1_onehot,
            hidden1_binary,
        )

    _, (loss, loss_binary, hidden1_onehot_all, hidden1_binary_all) = lax.scan(
        train_epoch,
        (train_state, train_state_binary, rng),
        (jnp.arange(num_epochs)),
        num_epochs,
    )

    # Extract activations for all epochs
    activations_dict = {}
    labels_single = jnp.array([0, 1])  # Labels for the two single examples
    for epoch in np.arange(num_epochs):
        activations_dict[epoch] = {
            "onehot": hidden1_onehot_all[epoch],
            "binary": hidden1_binary_all[epoch],
            "labels": labels_single,
        }

    return loss, loss_binary, activations_dict


# run - use multiple seeds for averaging
rng = jax.random.PRNGKey(0)
rngs = jax.random.split(rng, num_seeds)
# Run training for all seeds (can't use jit with dictionaries)
loss_all, loss_binary_all, activations_dicts_all = jax.vmap(train)(rngs)

# activations_dicts_all is a pytree: dict with keys being epochs, values are dicts with 'onehot', 'binary', 'labels'
# Each value is vmapped, so activations_dicts_all[epoch]['onehot'] has shape (num_seeds, ...)
# Convert to list of dictionaries for easier access
activations_dicts_list = []
for seed_idx in range(num_seeds):
    seed_dict = {}
    for epoch in np.arange(num_epochs):
        if epoch in activations_dicts_all:
            seed_dict[epoch] = {
                "onehot": activations_dicts_all[epoch]["onehot"][seed_idx],
                "binary": activations_dicts_all[epoch]["binary"][seed_idx],
                "labels": (
                    activations_dicts_all[epoch]["labels"][seed_idx]
                    if activations_dicts_all[epoch]["labels"].ndim > 0
                    else activations_dicts_all[epoch]["labels"]
                ),
            }
    activations_dicts_list.append(seed_dict)

# Use first seed for visualization (single example)
loss, loss_binary, activations_dict = (
    loss_all[0],
    loss_binary_all[0],
    activations_dicts_list[0],
)


# plot results - train loss over epochs
loss_mean = loss_all.mean(axis=0)
loss_binary_mean = loss_binary_all.mean(axis=0)

loss_se = loss_all.std(axis=0) / np.sqrt(num_seeds)
loss_binary_se = loss_binary_all.std(axis=0) / np.sqrt(num_seeds)


# ============================================================================
# Separability Visualization
# ============================================================================


def compute_separability_metrics(activations, labels):
    """Compute inter-class distance and cosine metrics (for single examples)"""
    activations = np.array(activations)
    labels = np.array(labels).flatten()

    # Get class indices
    class0_mask = labels == 0
    class1_mask = labels == 1

    class0_acts = activations[class0_mask]
    class1_acts = activations[class1_mask]

    vec0 = class0_acts[0]
    vec1 = class1_acts[0]

    # Euclidean distance
    euclidean_dist = np.linalg.norm(vec0 - vec1)

    # Cosine similarity and distance
    dot_product = np.dot(vec0, vec1)
    norm0 = np.linalg.norm(vec0)
    norm1 = np.linalg.norm(vec1)

    # Handle zero vectors
    if norm0 == 0 or norm1 == 0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = dot_product / (norm0 * norm1)

    cosine_distance = 1 - cosine_similarity

    return {
        "euclidean_dist": euclidean_dist,
        "cosine_distance": cosine_distance,
        "cosine_similarity": cosine_similarity,
        "separability_ratio": cosine_distance,  # Use cosine distance as separability metric
    }


# Collect cosine similarity metrics over epochs (averaged across all seeds)
cosine_sim_onehot = []
cosine_sim_binary = []
cosine_sim_onehot_se = []
cosine_sim_binary_se = []

for epoch in np.arange(num_epochs):
    # Collect metrics from all seeds
    cosine_sim_onehot_all = []
    cosine_sim_binary_all = []

    for seed_idx in range(num_seeds):
        seed_dict = activations_dicts_list[seed_idx]
        if epoch in seed_dict:
            data_onehot = np.array(seed_dict[epoch]["onehot"])
            data_binary = np.array(seed_dict[epoch]["binary"])
            labels = np.array(seed_dict[epoch]["labels"])

            metrics_onehot = compute_separability_metrics(data_onehot, labels)
            metrics_binary = compute_separability_metrics(data_binary, labels)

            cosine_sim_onehot_all.append(metrics_onehot["cosine_similarity"])
            cosine_sim_binary_all.append(metrics_binary["cosine_similarity"])

    if cosine_sim_onehot_all:
        cosine_sim_onehot.append(np.mean(cosine_sim_onehot_all))
        cosine_sim_binary.append(np.mean(cosine_sim_binary_all))
        cosine_sim_onehot_se.append(
            np.std(cosine_sim_onehot_all) / np.sqrt(len(cosine_sim_onehot_all))
        )
        cosine_sim_binary_se.append(
            np.std(cosine_sim_binary_all) / np.sqrt(len(cosine_sim_binary_all))
        )

epochs_sorted = np.arange(num_epochs)[: len(cosine_sim_onehot)]

# Define colors
color_binary = "#E69F00"
color_onehot = "#40B0A6"

# Create training loss plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    np.arange(num_epochs),
    loss_mean,
    "D-",
    label="Binary Encoding [00/11]",
    linewidth=2,
    markersize=10,
    color=color_binary,
)
ax.fill_between(
    np.arange(num_epochs),
    loss_mean - loss_se,
    loss_mean + loss_se,
    alpha=0.2,
    color=color_binary,
)
ax.plot(
    np.arange(num_epochs),
    loss_binary_mean,
    "D-",
    label="One-Hot Encoding [10/01]",
    linewidth=2,
    markersize=10,
    color=color_onehot,
)
ax.fill_between(
    np.arange(num_epochs),
    loss_binary_mean - loss_binary_se,
    loss_binary_mean + loss_binary_se,
    alpha=0.2,
    color=color_onehot,
)
ax.set_xlabel("Epoch", fontsize=16)
ax.set_ylabel("Train loss", fontsize=16)
ax.set_ylim(-0.01, None)
ax.set_xlim(0, num_epochs - 1)
ax.set_xticks(np.arange(0, num_epochs))
ax.tick_params(axis="both", which="major", labelsize=16)  # Major tick labels
ax.legend(fontsize=16)
# ax.grid(True, alpha=0.3)
# ---- Grid lines (y-axis only) ----
major_width = 1
minor_width = 0.5
major_alpha = 0.5
minor_alpha = 0.3
ax.grid(
    True,
    which="major",
    axis="y",
    linewidth=major_width,
    color="gray",
    alpha=major_alpha,
)
for spine in ax.spines.values():
    spine.set_edgecolor("gray")
    spine.set_linewidth(1.0)

fig.set_size_inches(16 / 2.54, 12 / 2.54)  # Set figure size to 20cm by 20cm
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "training_loss_over_epochs.jpg"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "training_loss_over_epochs.pdf"),
    bbox_inches="tight",
)

print("Saved training loss plot to training_loss_over_epochs.jpg")

# Create cosine similarity plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    epochs_sorted,
    cosine_sim_onehot,
    "D-",
    label="One-Hot Encoding [10/01]",
    linewidth=2,
    markersize=10,
    color=color_onehot,
)
ax.fill_between(
    epochs_sorted,
    np.array(cosine_sim_onehot) - np.array(cosine_sim_onehot_se),
    np.array(cosine_sim_onehot) + np.array(cosine_sim_onehot_se),
    alpha=0.2,
    color=color_onehot,
)
ax.plot(
    epochs_sorted,
    cosine_sim_binary,
    "D-",
    label="Binary Encoding [00/11]",
    linewidth=2,
    markersize=10,
    color=color_binary,
)
ax.fill_between(
    epochs_sorted,
    np.array(cosine_sim_binary) - np.array(cosine_sim_binary_se),
    np.array(cosine_sim_binary) + np.array(cosine_sim_binary_se),
    alpha=0.2,
    color=color_binary,
)
ax.set_xlabel("Epoch", fontsize=16)
ax.set_ylabel("Cosine Similarity", fontsize=16)
ax.set_xlim(min(epochs_sorted), max(epochs_sorted))
ax.set_xticks(epochs_sorted)
ax.tick_params(axis="both", which="major", labelsize=16)  # Major tick labels
# ax.legend(fontsize=16)
# ---- Grid lines (y-axis only) ----
major_width = 1
minor_width = 0.5
major_alpha = 0.5
minor_alpha = 0.3
ax.grid(
    True,
    which="major",
    axis="y",
    linewidth=major_width,
    color="gray",
    alpha=major_alpha,
)
for spine in ax.spines.values():
    spine.set_edgecolor("gray")
    spine.set_linewidth(1.0)

fig.set_size_inches(16 / 2.54, 12 / 2.54)  # Set figure size to 20cm by 20cm
plt.tight_layout()
plt.savefig(
    os.path.join(os.path.dirname(__file__), "cosine_similarity_over_training.jpg"),
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(os.path.dirname(__file__), "cosine_similarity_over_training.pdf"),
    bbox_inches="tight",
)
print("Saved cosine similarity plot to cosine_similarity_over_training.jpg")
