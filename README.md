## Repo for Deep Reinforcement Learning Project
This repo contains the code for my Deep Reinforcement Project: An Exploration of Q-Learning using Normalized Advantage Functions and Model-Based Accelerations in Various Different Domains

The `runner.py` file will handle all putting together the different pieces to run the various algorithms. **This requires 3 flags in a certain order: env, alg, custom_reward**. The `test.sh` contains examples of how to run the code from the command line.

The code for this project uses mainly uses:
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
- [An implementation of NAF](https://github.com/BY571/Normalized-Advantage-Function-NAF-) by Dittert, Sebastian. Code is in the naf/ directory.
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)

Modifications:
- I modified my pip installation of Stable Baselines 3 such that the TD3 MLP policy includes BatchNorms after each hidden layer. Unfortunately I'm not sure how to indicate which files were modified where, so runner beware.
- I modified the NAF implementation to add iLQG and imagination rollouts for model-based accelerations

