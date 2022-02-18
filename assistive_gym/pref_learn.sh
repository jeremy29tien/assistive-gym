#!/bin/bash
#Note that this will only work for our lab Linux machines, since my Mac runs zsh

seed=0
for seed in 0 1 2; do
  echo Seed $seed

  #Reward-learning
  config="raw_states/100prefs_100epochs_10patience_001lr_01weightdecay"
  reward_model_path="trex/models/${config}_seed${seed}.params"
  reward_output_path="trex/reward_learning_outputs/${config}_seed${seed}.txt"

  python3 trex/model.py --num_comps 100 --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.1 --seed $seed --test --reward_model_path $reward_model_path > $reward_output_path

  #RL
  policy_save_dir="./trained_models_reward_learning/${config}_seed${seed}"
  python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --save-dir $policy_save_dir

  #Eval
  load_policy_path="${policy_save_dir}/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_53/checkpoint-53"
  eval_path="trex/rl/eval/${config}_seed${seed}.txt"
  python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $eval_path
done


