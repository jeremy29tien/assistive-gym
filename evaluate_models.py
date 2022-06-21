from gpu_utils import determine_default_torch_device
from trex.model import create_training_data, Net, calc_accuracy
import argparse
import numpy as np
import torch


def evaluate_models(infile, test_data_dir, outdir):
    with open(infile) as f:
        model_paths = f.readlines()
    model_paths = [s.strip() for s in model_paths]

    demos = np.load(test_data_dir + "/demos.npy")
    demo_rewards = np.load(test_data_dir + "/demo_rewards.npy")
    demo_reward_per_timestep = np.load(test_data_dir + "/demo_reward_per_timestep.npy")
    sorted_demos = np.array([x for _, x in sorted(zip(demo_rewards, demos), key=lambda pair: pair[0])])
    sorted_rewards = np.array(sorted(demo_rewards))

    test_inputs, test_outputs = create_training_data(sorted_demos, sorted_rewards, all_pairs=True)

    test_accuracies = []
    for model_path in model_paths:
        if model_path:
            model = Net("feeding" if "feeding" in model_path else "scratch_itch", hidden_dims=(128, 64), fully_observable=True)
            model.load_state_dict(torch.load(model_path))
            device = torch.device(determine_default_torch_device(not torch.cuda.is_available()))
            model.to(device)

            acc = calc_accuracy(device, model, test_inputs, test_outputs)
            test_accuracies.append(acc)

    np.save(outdir + "/test_accuracies.npy", np.asarray(test_accuracies))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--infile', default='',
                        help='Input file with model paths.')
    parser.add_argument('--test_data_dir', default='', help='path to test data.')
    parser.add_argument('--outdir', default='',
                        help='Output directory.')

    args = parser.parse_args()

    evaluate_models(args.infile, args.test_data_dir, args.outdir)
