Trained:
1. Set FeedingLinearRewardEnv's `reward_net_path` to `/home/jtien/assistive-gym/trex/models/handpicked/5000traj_1epoch_noweightdecay_earlystopping.params`
2. Run `python3 -m assistive_gym.learn --env "FeedingLinearRewardSawyer-v0" --algo ppo --seed 0 --train --train-timesteps 1000000 --save-dir ./trained_models_reward_learning/linear/noweightdecay`

Evaluated:
- On linear reward: python3 -m assistive_gym.learn --env "FeedingLinearRewardSawyer-v0" --algo ppo --evaluate --eval-episodes 100 --seed 1 --verbose --load-policy-path  ./trained_models_reward_learning/linear/noweightdecay/ppo/FeedingLinearRewardSawyer-v0/checkpoint_53/checkpoint-53 > trex/rl/eval/linear/noweightdecay/learnedpolicy_linearreward.txt
- On ground truth reward: python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 1 --verbose --load-policy-path  ./trained_models_reward_learning/linear/noweightdecay/ppo/FeedingLinearRewardSawyer-v0/checkpoint_53/checkpoint-53 > trex/rl/eval/linear/noweightdecay/learnedpolicy_truereward.txt

