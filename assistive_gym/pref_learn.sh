#!/bin/bash
#Note that this will only work for our lab Linux machines, since my Mac runs zsh

var1=$1  # This is the variable that will hold our independent variable in our experiments
var2=$2
for seed in 0 1 2; do
  echo Seed $seed

  #Reward-learning
  echo "Reward learning..."
  if [ "$var1" = "--teleop" ]; then
      mode="teleop"
    else
      mode="rlnoise"
  fi
  config="data_experiment/${mode}_augmented10raw_10demosallpairs_100epochs_10patience_001lr_001weightdecay_001l1reg"
  reward_model_path="/home/jeremy/assistive-gym/trex/models/${config}_seed${seed}.params"
  reward_output_path="reward_learning_outputs/${config}_seed${seed}.txt"

  cd trex/
  python3 model.py ${var1} --augmented --num_rawfeatures 10 --num_demos 10 --all_pairs --num_epochs 100 --patience 10 --lr 0.01 --weight_decay 0.01 --l1_reg 0.01 --seed $seed --reward_model_path $reward_model_path > $reward_output_path

  #RL
  echo "Performing RL..."
  cd ..
  policy_save_dir="./trained_models_reward_learning/${config}_seed${seed}"
  python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --save-dir $policy_save_dir

  #Eval
  echo "Evaluating RL..."
  load_policy_path="${policy_save_dir}/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_000053/checkpoint-53"
  eval_path="trex/rl/eval/${config}_seed${seed}.txt"
  python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $eval_path
done



