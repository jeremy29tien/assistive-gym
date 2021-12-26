### Reward Learning
5 demos, all pairs, 10 epochs, learning rate = 0.01, weight decay = 0.1, no bias.
``` 
python3 linear_model.py --num_demos 5 --all_pairs --num_epochs 10 --lr 0.01 --weight_decay 0.1 --seed 2 --reward_model_path models/linear/5demosallpairs_10epochs_001lr_01weightdecay_seed2.params > reward_learning_outputs/linear/5demosallpairs_10epochs_001lr_01weightdecay_seed2.txt
```

### RL Training
- Modify `FeedingLinearRewardEnv`'s path to use `/home/jtien/assistive-gym/trex/models/linear/5demosallpairs_10epochs_001lr_01weightdecay_seed2.params`
- Run
	```
	python3 -m assistive_gym.learn --env "FeedingLinearRewardSawyer-v0" --algo ppo --seed 2 --train --train-timesteps 1000000 --save-dir ./trained_models_reward_learning/linear/weightdecay/stress_test/5demosallpairs_10epochs_001lr_01weightdecay_seed2
	```

### RL Evaluation
- Modify `FeedingLinearRewardEnv`'s path to use `/home/jtien/assistive-gym/trex/models/linear/5demosallpairs_10epochs_001lr_01weightdecay_seed2.params`
- Evaluate (seed=3) on ground truth reward:
    ```
    python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path  ./trained_models_reward_learning/linear/weightdecay/stress_test/5demosallpairs_10epochs_001lr_01weightdecay_seed2/ppo/FeedingLinearRewardSawyer-v0/checkpoint_53/checkpoint-53 > trex/rl/eval/linear/weightdecay/stress_test/5demosallpairs_10epochs_001lr_01weightdecay_seed2/learnedpolicy_truereward.txt
    ```
- Evaluate (seed=3) on learned reward:
    ```
    python3 -m assistive_gym.learn --env "FeedingLinearRewardSawyer-v0" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path  ./trained_models_reward_learning/linear/weightdecay/stress_test/5demosallpairs_10epochs_001lr_01weightdecay_seed2/ppo/FeedingLinearRewardSawyer-v0/checkpoint_53/checkpoint-53 > trex/rl/eval/linear/weightdecay/stress_test/5demosallpairs_10epochs_001lr_01weightdecay_seed2/learnedpolicy_learnedreward.txt
    ```

