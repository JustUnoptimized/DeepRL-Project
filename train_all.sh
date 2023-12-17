# DDPG
# sbatch ./train.slurm runner.py --env Pendulum-v1 --alg ddpg --custom_reward default --total_timesteps 40000 --verbose 1 --log_interval 100 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env MountainCarContinuous-v0 --alg ddpg --custom_reward default --total_timesteps 40000 --verbose 1 --log_interval 100 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward default --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward simple --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward magni --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward kovatchev --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5

# TD3
# sbatch ./train.slurm runner.py --env Pendulum-v1 --alg td3 --custom_reward default --total_timesteps 40000 --verbose 1 --log_interval 100 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env MountainCarContinuous-v0 --alg td3 --custom_reward default --total_timesteps 40000 --verbose 1 --log_interval 100 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward default --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward simple --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward magni --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward kovatchev --total_timesteps 200000 --verbose 1 --log_interval 1000 --eval_freq 5000 --n_eval_episodes 5

# NAF
# sbatch ./train.slurm runner.py --env Pendulum-v1 --alg naf --custom_reward default --total_timesteps 40000 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env MountainCarContinuous-v0 --alg naf --custom_reward default --total_timesteps 40000 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward default --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward simple --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward magni --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward kovatchev --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5

# # NAF + iLQG
# sbatch ./train.slurm runner.py --env Pendulum-v1 --alg naf-mba --custom_reward default --refit_every 5 --mu_prob 0.8 --total_timesteps 40000 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env MountainCarContinuous-v0 --alg naf-mba --custom_reward default --refit_every 5 --mu_prob 0.8 --total_timesteps 40000 --eval_freq 1000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward default --refit_every 5 --mu_prob 0.8 --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward simple --refit_every 5 --mu_prob 0.8 --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward magni --refit_every 5 --mu_prob 0.8 --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5
# sbatch ./train.slurm runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward kovatchev --refit_every 5 --mu_prob 0.8 --total_timesteps 200000 --eval_freq 5000 --n_eval_episodes 5