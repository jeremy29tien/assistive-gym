import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from trex.model import Net


def compute_saliency_maps(X, model):
    """
    Compute a class saliency map using the model for trajectory X.

    Input:
    - X: Input trajectory; Tensor of shape (T, D), where T is length of trajectory and D is dimension of feature space.
    - model: A pretrained model that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (D, ) giving the saliency maps for the input (we will max over the time dimension).
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Construct new tensor that requires gradient computation
    X = X.clone().detach().requires_grad_(True)
    saliency = None

    T = X.shape[0]

    # Forward pass
    cum_reward = model.cum_return(X)

    # Compute loss.
    # Essentially, we're using the model-outputted score of the correct class as the loss.
    # We're not necessarily interested in minimizing this value per se; we're more interested
    # in the size of the (large or small) gradients that we use when we're backpropagating.
    loss = cum_reward

    # Backward pass
    loss.backward()

    g = X.grad

    # dims will be (D, ) after maxing across T dimension.
    saliency = torch.max(torch.abs(g), dim=0)[0]  # torch.max() returns a tuple of (values, indices)
    saliency_per_timestep = torch.abs(g)
    grad_per_timestep = g

    return saliency, saliency_per_timestep, grad_per_timestep


def load_model(path):
    model = Net("feeding", hidden_dims=(128, 64), fully_observable=True)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', default='', help='Path to saved model file.')
    args = parser.parse_args()

    demos = np.load("trex/data/feeding/fully_observable/demos.npy")
    rewards = np.load("trex/data/feeding/fully_observable/demo_rewards.npy")
    X = demos[0]
    print("reward of X:", rewards[0])
    X = torch.from_numpy(X).float()

    model = load_model(args.model)
    saliency_map, saliency_per_timestep, grad_per_timestep = compute_saliency_maps(X, model)

    plt.title("Saliency (max across timesteps of absolute values)")
    plt.xlabel("feature")
    plt.imshow(saliency_map.reshape((1, 40)), cmap=plt.cm.hot)
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(saliency_per_timestep, cmap=plt.cm.hot)
    # set aspect ratio to 1
    ratio = 2.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    plt.title("Saliency per timestep (absolute values)")
    plt.ylabel("timestep")
    plt.xlabel("feature")
    plt.show()

    fig, ax = plt.subplots()
    ax.imshow(grad_per_timestep, cmap=plt.cm.hot)
    # set aspect ratio to 1
    ratio = 2.0
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    plt.title("Gradient per timestep")
    plt.ylabel("timestep")
    plt.xlabel("feature")
    plt.show()


