#!/bin/bash
#Note that this will only work for our lab Linux machines, since my Mac runs zsh

var1=$1  # This is the variable that will hold our independent variable in our experiments
var2=$2
for seed in 0 1 2; do
  echo Seed $seed

  #Reward-learning
  echo "Reward learning..."
  config="feeding/vanilla/${var2}${var1}_324demos_allpairs_hdim256-256-256_100epochs_10patience_00001lr_0000001weightdecay"

  reward_model_path="/home/jeremy/assistive-gym/trex/models/${config}_seed${seed}.params"
  reward_output_path="reward_learning_outputs/${config}_seed${seed}.txt"

  cd trex/
  if ! [[ $seed -eq 0 ]]
  then
    python3 model.py --feeding --seed $seed --$var1 $var2 --fully_observable --hidden_dims 256 256 256 --num_demos 324 --all_pairs --num_epochs 100 --patience 10 --lr 0.0001 --weight_decay 0.000001 --reward_model_path $reward_model_path > $reward_output_path
  fi

  #RL
  echo "Performing RL..."
  cd ..
  policy_save_dir="./trained_models_reward_learning/${config}_seed${seed}"
  python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo ppo --seed $seed --train --train-timesteps 1000000 --reward-net-path $reward_model_path --save-dir $policy_save_dir --load-policy-path $policy_save_dir --tb

  #Eval
  echo "Evaluating RL..."
  load_policy_path="${policy_save_dir}/ppo/FeedingLearnedRewardSawyer-v0/checkpoint_40/checkpoint-40"
  gt_eval_path="trex/rl/eval/${config}_seed${seed}.txt"
  learned_eval_path="trex/rl/eval/${config}_seed${seed}_learnedreward.txt"
  python3 -m assistive_gym.learn --env "FeedingSawyer-v1" --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $gt_eval_path
  python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --reward-net-path $reward_model_path --algo ppo --evaluate --eval-episodes 100 --seed 3 --verbose --load-policy-path $load_policy_path > $learned_eval_path
done



