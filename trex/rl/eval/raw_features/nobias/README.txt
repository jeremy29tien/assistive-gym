TREX training: 
python3 model.py --num_trajs 5000 --num_epochs 100 --lr 0.00005 --no_bias --reward_model_path models/5000traj_100epoch_nobias_earlystopping.params > reward_learning_outputs/raw_features/5000traj_100epoch_nobias_earlystopping.txt

Running RL: 
1. Modify `FeedingLearnedRewardEnv`'s reward_net_path to /home/jtien/assistive-gym/trex/models/5000traj_100epoch_nobias_earlystopping.params and set `with_bias=False`
2. Run `python3 -m assistive_gym.learn --env "FeedingLearnedRewardSawyer-v0" --algo ppo --seed 0 --train --train-timesteps 1000000 --save-dir ./trained_models_reward_learning/nobias`
