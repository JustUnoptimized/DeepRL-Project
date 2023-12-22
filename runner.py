# basic imports
from argparse import ArgumentParser
import os
import os.path as osp
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time

# gymnasium imports
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility

# NAF imports
from naf.agent import NAF_Agent
from naf.naf import naf_runner, naf_final_eval

# DDPG and TD3 imports
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Common util imports
from commonutils import print_timedelta, get_horizon, register_simglucose

# Evaluation imports
from evalutils import training_eval_plots, do_rollouts, final_eval_plot

############################################
import warnings
warnings.filterwarnings("ignore")
############################################


if __name__ == '__main__':
    print('Starting...')
    
    parser = ArgumentParser()
    
    envchoices = [
        'Pendulum-v1',
        'MountainCarContinuous-v0',
        'simglucose-adolescent2-v0'
    ]
    parser.add_argument('--env', required=True, type=str, choices=envchoices)
    
    algchoices = [
        'ddpg',
        'td3',
        'naf',
        'naf-mba'
    ]
    parser.add_argument('--alg', required=True, type=str, choices=algchoices)
    
    cust_rew_choices = [
        'default',
        'simple',
        'magni',
        'kovatchev'
    ]
    parser.add_argument('--custom_reward', '-r', required=True, type=str, choices=cust_rew_choices)
    parser.add_argument('--exp_name', required=True, type=str)
    
    # if flag set, only load models and do final evals. Check relevant args in final eval args section
    parser.add_argument('--only_final_eval', action='store_true')
    
    # noise args
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--sigma', type=float, default=0.3)
    
    # Common training args
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--learning_starts', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--total_timesteps', type=int, default=10000)
    
    # DDPG / TD3 training args
    parser.add_argument('--nobn', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--target_policy_noise', type=float, default=0.2)
    parser.add_argument('--target_noise_clip', type=float, default=0.5)
    
    # NAF + iLQG
    parser.add_argument('--mba', action='store_true')  # DON'T TOUCH! Specify using '--alg naf-mba'
    parser.add_argument('--refit_every', type=int, default=5)
    parser.add_argument('--mu_prob', type=float, default=0.5)
    parser.add_argument('--imag_rollouts', type=int, default=5)
    parser.add_argument('--imag_rollout_len', type=int, default=20)
    parser.add_argument('--default_lindyn', type=str, default='None')
    parser.add_argument('--lindyn_scale', type=float, default=1.)
    parser.add_argument('--rxu_quad_scale', type=float, default=1.)
    parser.add_argument('--rxu_lin_scale', type=float, default=1.)
    
    # don't touch these!!
    parser.add_argument('-f', "--frames", type=int, default=40000,
                     help='Number of training frames (default: 40000)')  # this is overwritten by total_timesteps!
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                     help='Learning rate (default: 1e-3)')
    parser.add_argument('-mem', type=int, default=1000000,
                     help='Replay buffer size (default: 100000)')
    parser.add_argument('-per', type=int, choices=[0,1],  default=0,
                     help='Use prioritized experience replay (default: False)')
    parser.add_argument("-d2rl", type=int, choices=[0,1], default=0,
                     help="Using D2RL Deep Dense NN Architecture if set to 1 (default: 0)")
    parser.add_argument('-l', "--layer_size", type=int, default=256,
                     help='Neural Network layer size (default: 256)')
    parser.add_argument("--clip_grad", type=float, default=1.0,
                     help="Clip gradients (default: 1.0)")
    parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse",
                     help="Choose loss type MSE or Huber loss (default: mse)")
    parser.add_argument('-u', "--update_every", type=int, default=1,
                     help='update the network every x step (default: 1)')
    parser.add_argument('-n_up', "--n_updates", type=int, default=1,
                     help='update the network for x steps (default: 1)')
    parser.add_argument('-nstep', type=int, default=1,
                     help='nstep_bootstrapping (default: 1)')
    
    # eval args
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--n_eval_episodes', type=int, default=5)

    # final eval args
    parser.add_argument('--n_final_eval', type=int, default=1)
    
    # other
    parser.add_argument('--nogpu', action='store_true')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    
    # hacky way to get compatibility between ddpg/td3 args and naf args
    args.frames = args.total_timesteps
    args.learning_rate = args.lr
    
    for k, v in vars(args).items():
        print(f'{k:<20s} : {v}')
        
    # set up relevant variables from cmd line arguments
    device = 'cpu' if args.nogpu else 'auto'
    
    if args.env == 'simglucose-adolescent2-v0':
        register_simglucose(custom_rew=args.custom_reward)
    
    env = gym.make(args.env)
    env.np_random = np.random.default_rng(seed=args.seed)
    
    eval_env = gym.make(args.env)  # for evalutions during training
    eval_env.np_random = np.random.default_rng(seed=args.seed+1)
    
    final_env = gym.make(args.env)  # for final evalution after training
    final_env.np_random = np.random.default_rng(seed=args.seed+2)
    final_eval_seed = args.seed + 2

    horizon = get_horizon(args.env)
    
    policy = 'MlpPolicy'

    if args.noise:
        du = env.action_space.shape[-1]  # shape of action space for added exploration noise
        action_noise = NormalActionNoise(mean=np.zeros(du), sigma=args.sigma * np.ones(du))
    else:
        action_noise = None
        
    if args.alg == 'naf-mba':
        args.mba = True
        
    # exp_name = f'{args.alg}_{args.env}_{args.custom_reward}'
    exp_name = args.exp_name
    outdir = osp.join('evals/', exp_name)
    if not osp.exists(outdir):
        os.mkdir(outdir)
    modelname = osp.join(outdir, 'model')
    
    policy_kwargs = {
        'net_arch': [256, 256],
        'nobn': args.nobn
    }
    
    # do training OR load trained model
    if args.alg == 'ddpg':
        # eval env needs to be the same type + wrappers as env
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # final env needs to be the same type + wrappers as env
        final_env = Monitor(final_env)
        final_env = DummyVecEnv([lambda: final_env])
        
        eval_cb = EvalCallback(eval_env,
                               n_eval_episodes=args.n_eval_episodes,
                               eval_freq=args.eval_freq,
                               log_path=outdir)
        
        model = DDPG(policy,
                     env,
                     learning_rate=args.lr,
                     buffer_size=1000000,
                     learning_starts=args.learning_starts,
                     batch_size=args.batch_size,
                     tau=args.tau,
                     gamma=args.gamma,
                     train_freq=(1, 'episode'),
                     gradient_steps=-1,
                     action_noise=action_noise,
                     replay_buffer_class=None,
                     replay_buffer_kwargs=None,
                     optimize_memory_usage=False,
                     tensorboard_log=None,
                     policy_kwargs=policy_kwargs,
                     verbose=args.verbose,
                     seed=args.seed,
                     device=device,
                     _init_setup_model=True)
        
        print(f'model.device = {model.device}')
        print(model.policy)
        
        if args.only_final_eval:
            # only do final evaluation
            print('Only doing final evaluation...')
            print('Loading model...')
            model = DDPG.load(modelname)
            
        else:
            print('Starting training...')
            t_start = time()
            model.learn(total_timesteps=args.total_timesteps,
                        log_interval=args.log_interval,
                        callback=eval_cb)
            t_end = time()
            print_timedelta(t_start, t_end, 'Training')
            print('Saving model...')
            model.save(modelname)

    elif args.alg == 'td3':
        # eval env needs to be the same type + wrappers as env
        eval_env = Monitor(eval_env)
        eval_env = DummyVecEnv([lambda: eval_env])
        
        # final env needs to be the same type + wrappers as env
        final_env = Monitor(final_env)
        final_env = DummyVecEnv([lambda: final_env])
        
        eval_cb = EvalCallback(eval_env,
                               n_eval_episodes=args.n_eval_episodes,
                               eval_freq=args.eval_freq,
                               log_path=outdir)
        
        model = TD3(policy,
                    env,
                    learning_rate=args.lr,
                    buffer_size=1000000,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    gamma=args.gamma,
                    train_freq=(1, 'episode'),
                    gradient_steps=-1,
                    action_noise=action_noise,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    policy_delay=2,
                    target_policy_noise=args.target_policy_noise,
                    target_noise_clip=args.target_noise_clip,
                    stats_window_size=100,
                    tensorboard_log=None,
                    policy_kwargs=policy_kwargs,
                    verbose=args.verbose,
                    seed=args.seed,
                    device=device,
                    _init_setup_model=True)

        print(f'model.device = {model.device}')
        print(model.policy)
        
        if args.only_final_eval:
            # only do final evaluation
            print('Only doing final evaluation...')
            print('Loading model...')
            model = TD3.load(modelname)
        
        else:
            print('Starting training...')
            t_start = time()
            model.learn(total_timesteps=args.total_timesteps,
                        log_interval=args.log_interval,
                        callback=eval_cb)
            t_end = time()
            print_timedelta(t_start, t_end, 'Training')
            print('Saving model...')
            model.save(modelname)

    elif args.alg == 'naf' or args.alg == 'naf-mba':
        from torch.utils.tensorboard import SummaryWriter
    
        # naf code uses old gym api where env.step(action) returns (next_state, reward, done, info)
        env = StepAPICompatibility(env, output_truncation_bool=False)
        eval_env = StepAPICompatibility(eval_env, output_truncation_bool=False)
        final_env = StepAPICompatibility(final_env, output_truncation_bool=False)

        action_size = env.action_space.shape[0]
        state_size = env.observation_space.shape[0]
        action_space = env.action_space
        state_space = env.observation_space
        
        # re-encode device keyword for naf code
        if device == 'auto':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        writer = SummaryWriter(f'runs/{exp_name}')

        model = NAF_Agent(state_size=state_size,
                          state_space=state_space,
                          action_size=action_size,
                          action_space=action_space,
                          horizon=horizon,
                          device=device, 
                          args=args,
                          writer=writer)
        
        print(f'model.device = {model.device}')
        print(model.qnetwork_local)
        
        if args.only_final_eval:
            # only do final evaluation
            print('Only doing final evaluation...')
            print('Loading model...')
            model.load_model(modelname)
            
        else:
            # do training
            print('Starting training...')
            t_start = time()
            naf_runner(args, model, env, eval_env, outdir, writer)
            t_end = time()
            print_timedelta(t_start, t_end, 'Training')
            print('Saving model...')
            torch.save(model.qnetwork_local.state_dict(), modelname + '_.pth')
            # save parameter
            with open('runs/'+exp_name+".json", 'w') as f:
                json.dump(args.__dict__, f, indent=2)
    
    # make plots
    if not args.only_final_eval:
        training_eval_plots(outdir, args.alg, args.env, args.n_eval_episodes)

    print(f'Doing {args.n_final_eval} rollouts...')
    obslists, actlists, rewlists = do_rollouts(args.alg, model, final_env, args.n_final_eval, final_eval_seed, maxiter=horizon)
    final_eval_plot(outdir, args.alg, args.env, obslists, actlists, rewlists)