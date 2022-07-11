import argparse
import torch
import numpy as np
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
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with torch.autograd.gard.           #
    ##############################################################################
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
    print(g.shape)

    # dims will be (D, ) after maxing across T dimension.
    saliency = torch.max(torch.abs(g), dim=0)[0]  # torch.max() returns a tuple of (values, indices)
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return saliency


def load_model(path):
    model = Net("feeding", hidden_dims=(128, 64), fully_observable=True)
    model.load_state_dict(torch.load(path))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--model', default='',
                        help='Path to saved model file.')
    args = parser.parse_args()

    X = np.load("trex/data/feeding/fully_observable/demos.npy")[0]

    model = load_model(args.model)
    saliency_map = compute_saliency_maps(X, model)

