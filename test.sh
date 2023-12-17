#### Training
## DDPG tests
# python runner.py --env Pendulum-v1 --alg ddpg --custom_reward default --total_timesteps 500 --verbose 1 --log_interval 25
# python runner.py --env MountainCarContinuous-v0 --alg ddpg --custom_reward default --total_timesteps 500 --verbose 1 --log_interval 25
# python runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward simple --total_timesteps 500 --verbose 1 --log_interval 25 --eval_freq 200 --n_eval_episodes 5

## TD3 tests
# python runner.py --env Pendulum-v1 --alg td3 --custom_reward default --total_timesteps 100 --verbose 1 --log_interval 25
# python runner.py --env MountainCarContinuous-v0 --alg td3 --custom_reward default --total_timesteps 100 --verbose 1 --log_interval 25
# python runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward simple --total_timesteps 500 --verbose 1 --log_interval 25 --eval_freq 200 --n_eval_episodes 5

## NAF tests
# python runner.py --env Pendulum-v1 --alg naf --custom_reward default --total_timesteps 5000 --eval_freq 100 --n_eval_episodes 5
# python runner.py --env MountainCarContinuous-v0 --alg naf --custom_reward default --total_timesteps 10000 --eval_freq 500 --n_eval_episodes 5
# python runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward simple --total_timesteps 500 --eval_freq 200 --n_eval_episodes 5

## NAF + iLQG tests
# python runner.py --env Pendulum-v1 --alg naf-mba --custom_reward default --refit_every 5 --mu_prob 0.8 --total_timesteps 2000 --eval_freq 100 --n_eval_episodes 5
# python runner.py --env MountainCarContinuous-v0 --alg naf-mba --custom_reward default --refit_every 5 --mu_prob 0.8 --total_timesteps 5000 --eval_freq 100 --n_eval_episodes 5
# python runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward simple --refit_every 5 --mu_prob 0.8 --total_timesteps 1000 --eval_freq 100 --n_eval_episodes 5


#### Only Final Eval
## DDPG final evals
# python runner.py --env Pendulum-v1 --alg ddpg --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env MountainCarContinuous-v0 --alg ddpg --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward kovatchev --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward magni --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg ddpg --custom_reward simple --only_final_eval --n_final_eval 9

# ## TD3 final evals
# python runner.py --env Pendulum-v1 --alg td3 --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env MountainCarContinuous-v0 --alg td3 --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward kovatchev --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward magni --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg td3 --custom_reward simple --only_final_eval --n_final_eval 9

# ## NAF final evals
# python runner.py --env Pendulum-v1 --alg naf --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env MountainCarContinuous-v0 --alg naf --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward kovatchev --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward magni --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf --custom_reward simple --only_final_eval --n_final_eval 9

# ## NAF + iLQG final evals
# python runner.py --env Pendulum-v1 --alg naf-mba --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env MountainCarContinuous-v0 --alg naf-mba --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward default --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward kovatchev --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward magni --only_final_eval --n_final_eval 9
# python runner.py --env simglucose-adolescent2-v0 --alg naf-mba --custom_reward simple --only_final_eval --n_final_eval 9
