### Reward Learning
- 6 demos, all pairs, 100 epochs, weight_decay=1.0
``` 
python3 linear_model.py --num_demos 6 --all_pairs --num_epochs 100 --weight_decay 1.0 --reward_model_path models/6demosallpairs_100epochs_1weightdecay.params > reward_learning_outputs/linear/6demosallpairs_100epochs_1weightdecay.txt
```

### RL Training
- Modify `FeedingLinearRewardEnv`'s path to use `/home/jtien/assistive-gym/trex/models/linear/6demosallpairs_100epochs_1weightdecay.params`
```
python3 -m assistive_gym.learn --env "FeedingLinearRewardSawyer-v0" --algo ppo --seed 0 --train --train-timesteps 1000000 --save-dir ./trained_models_reward_learning/linear/weightdecay/stress_test/6demos
```

### Evaluation
- Evaluate (seed=1) on ground truth reward:
	```
	python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 1 --verbose --load-policy-path  ./trained_models_reward_learning/linear/weightdecay/stress_test/6demos/ppo/FeedingLinearRewardSawyer-v0/checkpoint_53/checkpoint-53 > trex/rl/eval/linear/weightdecay/stress_test/learnedpolicy_truereward.txt
	```
- Evaluate (seed=1) on learned reward:
	```
	python3 -m assistive_gym.learn --env "FeedingLinearRewardSawyer-v0" --algo ppo --evaluate --eval-episodes 100 --seed 1 --verbose --load-policy-path  ./trained_models_reward_learning/linear/weightdecay/stress_test/6demos/ppo/FeedingLinearRewardSawyer-v0/checkpoint_53/checkpoint-53 > trex/rl/eval/linear/weightdecay/stress_test/learnedpolicy_learnedreward.txt
	```
