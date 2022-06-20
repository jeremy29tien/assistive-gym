import assistive_gym.learn
import argparse
import numpy as np


EVAL_SEED = 3


# infile is a text file with paths to trained policies
# separated by newline.
def evaluate_policies(infile, outdir):
    with open(infile) as f:
        policy_paths = f.readlines()
    policy_paths = [s.strip() for s in policy_paths]

    gt_reward_means = []
    slearned_reward_means = []
    mlearned_reward_means = []
    llearned_reward_means = []
    success_means = []
    for policy_path in policy_paths:
        if policy_path:
            # 1. Evaluate on the GT reward.
            reward_mean, reward_std, success_mean, success_std = assistive_gym.learn.evaluate_policy("FeedingSawyer-v1", "ppo", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False)
            gt_reward_means.append(reward_mean)
            success_means.append(success_mean)

            # 2. Evaluate on the learned reward(s)
            if 'seed0' in policy_path:
                seed = 0
            elif 'seed1' in policy_path:
                seed = 1
            elif 'seed2' in policy_path:
                seed = 2
            else:
                raise ValueError("Seed not specified.")
            sconfig = "scratch_itch/vanilla/40demos_hdim128-64_fullyobservable_allpairs_100epochs_10patience_0001lr_0001weightdecay"
            reward_model_path = "/home/jeremy/assistive-gym/trex/models/"+sconfig+"_seed"+str(seed)+".params"
            reward_mean, reward_std, _, _ = assistive_gym.learn.evaluate_policy("ScratchItchLearnedRewardJaco-v0", "ppo", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False, reward_net_path=reward_model_path)
            slearned_reward_means.append(reward_mean)

            mconfig = "scratch_itch/vanilla/120demos_hdim128-64_fullyobservable_allpairs_100epochs_10patience_0001lr_0001weightdecay"
            reward_model_path = "/home/jeremy/assistive-gym/trex/models/"+mconfig+"_seed"+str(seed)+".params"
            reward_mean, reward_std, _, _ = assistive_gym.learn.evaluate_policy("ScratchItchLearnedRewardJaco-v0", "ppo", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False, reward_net_path=reward_model_path)
            mlearned_reward_means.append(reward_mean)

            lconfig = "scratch_itch/vanilla/324demos_hdim128-64_fullyobservable_allpairs_100epochs_10patience_0001lr_0001weightdecay"
            reward_model_path = "/home/jeremy/assistive-gym/trex/models/"+lconfig+"_seed"+str(seed)+".params"
            reward_mean, reward_std, _, _ = assistive_gym.learn.evaluate_policy("ScratchItchLearnedRewardJaco-v0", "ppo", policy_path, n_episodes=100, seed=EVAL_SEED, verbose=False, reward_net_path=reward_model_path)
            llearned_reward_means.append(reward_mean)

    gt_reward_means = np.asarray(gt_reward_means)
    slearned_reward_means = np.asarray(slearned_reward_means)
    mlearned_reward_means = np.asarray(mlearned_reward_means)
    llearned_reward_means = np.asarray(llearned_reward_means)
    success_means = np.asarray(success_means)
    np.save(outdir + "/gtrewards.npy", gt_reward_means)
    np.save(outdir + "/40demos_hdim128-64_fullyobservable_allpairs_100epochs_10patience_0001lr_0001weightdecay_learnedrewards.npy", slearned_reward_means)
    np.save(outdir + "/120demos_hdim128-64_fullyobservable_allpairs_100epochs_10patience_0001lr_0001weightdecay_learedrewards.npy", mlearned_reward_means)
    np.save(outdir + "/324demos_hdim128-64_fullyobservable_allpairs_100epochs_10patience_0001lr_0001weightdecay_learnedrewards.npy", llearned_reward_means)
    np.save(outdir + "/success.npy", success_means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--infile', default='',
                        help='Input file with policy paths.')
    parser.add_argument('--outdir', default='',
                        help='Output directory.')

    args = parser.parse_args()

    evaluate_policies(args.infile, args.outdir)
