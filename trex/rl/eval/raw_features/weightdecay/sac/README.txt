TREX training: 
python3 model.py --num_trajs 5000 --num_epochs 100 --lr 0.00005 --weight_decay 1.0 --reward_model_path models/5000traj_100epoch_1weightdecay_earlystopping.params > reward_learning_outputs/raw_features/5000traj_100epoch_1weightdecay_earlystopping.txt

Running RL: 
1. Check that FeedingLearnedRewardEnv's reward_net_path uses 5000traj_100epoch_1weightdecay_earlystopping.params
2. Run `python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo sac --seed 0 --train --train-timesteps 1000000 --save-dir ./trained_models_reward_learning/weightdecay`
