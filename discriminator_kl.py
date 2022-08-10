# Function similar to active_learn's get_rollouts to get rollouts from trained policy

# Function that prepares the training data:
# - Squishes the rollouts / trajectories into just a bunch of states
# - Adds labels of 0 for reward-learning and 1 for policy training.

# Define discriminator model that classifies between states from reward-learning training data (0)
# and states from the trained policy (1) by minimizing cross-entropy loss.

if __name__ == '__main__':
    # Have an argument for the reward-learning training data file
    # Have an argument for the trained policy to pull rollouts from
    # Have an argument for how many trajectories / rollouts to pull from both data pools

    # Load data used in reward-learning

    # Get rollouts from trained policy

    # Prepare discriminator training data

    # Train discriminator

    # Report training and validation accuracy

    # Save model. In the future, load model if exists.

    # Inference:
    # For the states visited during reward-learning, calculate the average return/logit. --> $D_{KL}(p(x) || q(x))$

    # For the states visited during the policy rollout, calculate the average return/logit and NEGATE. --> $D_{KL}(q(x) || p(x))$
    pass
