import discriminator_kl
import argparse
import numpy as np
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--indvar_str', default='', help="")
    parser.add_argument('--indvar_num', default=0.0, type=float, help="")
    args = parser.parse_args()

    results = dict()
    results['train_accs'] = []
    results['val_accs'] = []
    results['dkl_pq'] = []
    results['dkl_qp'] = []
    results['symmetric_dkl'] = []
    for seed in [0, 1, 2]:
        env = "FeedingSawyer-v1"
        prefix = "/home/jeremy/assistive-gym/"

        reward_learning_data_path = prefix + "trex/data/feeding/fully_observable/demos.npy"
        config = "feeding/vanilla/324demos_allpairs_hdim256-256-256_100epochs_10patience_00001lr_0000001weightdecay"

        trained_policy_path = prefix + "trained_models_reward_learning/" + config + "_seed" + str(seed) + "/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_40/checkpoint-40"
        discriminator_model_path = prefix + "discriminator_kl_models/" + config + "_seed" + str(seed) + ".params"

        train_acc, val_acc, dkl_pq, dkl_qp = discriminator_kl.run(env, seed, reward_learning_data_path, trained_policy_path,
                                                                  num_trajs=50, fully_observable=True, pure_fully_observable=False,
                                                                  load_weights=True, discriminator_model_path=discriminator_model_path,
                                                                  num_epochs=100, hidden_dims=(128, 128, 128), lr=0.01,
                                                                  weight_decay=0.0001, l1_reg=0.0, patience=10)
        results['train_accs'].append(train_acc)
        results['val_accs'].append(val_acc)
        results['dkl_pq'].append(dkl_pq)
        results['dkl_qp'].append(dkl_qp)
        results['symmetric_dkl'].append(dkl_pq+dkl_qp)

    results['avg_symmetric_dkl'] = np.mean(results['symmetric_dkl'])

    outfile = prefix+"discriminator_kl_outputs/"+config+"_results.json"
    with open(outfile, 'w') as f:
        # indent=2 is not needed but makes the file human-readable
        # if the data is nested
        json.dump(results, f, indent=2)

