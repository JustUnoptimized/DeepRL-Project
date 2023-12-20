# basic imports
from argparse import ArgumentParser
import os
import os.path as osp
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from time import time

# gymnasium imports
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility

# simglucose imports
from simglucose.simulation.scenario_gen import RandomScenario
from bgrisks import simple_reward
from bgrisks import my_reward_closure

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

############################################
import warnings
warnings.filterwarnings("ignore")
############################################


def register_simglucose(custom_rew='default'):
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = RandomScenario(start_time=start_time, seed=1)
    
    sg_kwargs = {
        'patient_name': 'adolescent#002',
        'custom_scenario': meal_scenario
    }
    
    rew_fns = {
        'simple': simple_reward,
        'magni': my_reward_closure('magni'),
        'kovatchev': my_reward_closure('kovatchev')
    }
    
    if not custom_rew == 'default':
        print(f'Using custom reward {custom_rew} for Simglucose...')
        rew_fn = rew_fns[custom_rew]
        sg_kwargs.update({'reward_fun': rew_fn})
    else:
        print(f'Using default reward = risk[t-1] - risk[t] for Simglucose...')

    gym.register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimGymnaisumEnv',
        kwargs=sg_kwargs
    )


def print_timedelta(t_start, t_end, objective):
    h, rem = divmod(t_end - t_start, 3600)
    m, s = divmod(rem, 60)
    s = round(s)
    print(f'\n{objective} took {h} hours {m} minutes {s} seconds')


def get_horizon(envname):
    if envname == 'Pendulum-v1':
        horizon = 200
    elif envname == 'MountainCarContinuous-v0':
        horizon = 999
    elif envname == 'simglucose-adolescent2-v0':
        horizon = 999  # simglucose does not have specified time horizon, so arbitrarily choose 999
    
    return horizon
    
    
def training_eval_plots(outdir, n_eval_episodes):
    print('Plotting training evaluation rewards...')
    npzfile = np.load(osp.join(outdir, 'evaluations.npz'))
    timesteps = npzfile['timesteps']
    avgrewards = npzfile['results'].mean(axis=1)
    avgeplens = npzfile['ep_lengths'].mean(axis=1)
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Avg Reward and Episode Lengths over {n_eval_episodes} Trials')
    
    ax1.plot(avgrewards, color='red', label='Avg Reward')
    ax1.set_xticks(ticks=np.arange(timesteps.shape[0]), labels=timesteps, rotation=90, ha='right', fontsize=8)
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Avg Reward', color='red')
    
    ax2 = ax1.twinx()
    ax2.plot(avgeplens, color='gray', label='Avg Ep Length')
    ax2.set_ylabel('Avg Ep Length', color='gray')
    
    fig.tight_layout()
    plotname = 'training_eval.png'
    plotdir = osp.join(outdir, plotname)
    plt.savefig(plotdir)
    plt.close()
    

def do_rollouts(model, alg, env, n_final_eval, maxiter=1000):
    obslist = []
    actlist = []
    rewlist = []
    
    for k in range(n_final_eval):
        obslist_i = []
        # this funky thing is just so all the elements of actlist have the same shape
        # which is equal to env.action_space.shape -- I am assuming this is a 1D box
        actlist_i = [np.array([np.nan for _ in range(env.action_space.shape[0])])]
        rewlist_i = [np.nan]

        obs = env.reset()
        obslist_i.append(obs[0])
        for i in range(maxiter):
            action, _states = model.predict(obs)  # get action from policy
            obs, rewards, dones, info = env.step(action)
            done = dones[0]
            obslist_i.append(obs[0])
            actlist_i.append(action[0])
            rewlist_i.append(rewards[0])
            if done or i > maxiter:
                print(f'Rollout {k+1:02d} was {len(obslist_i)-1} actions long...')
                break
        
        obslist.append(obslist_i)
        actlist.append(actlist_i)
        rewlist.append(rewlist_i)

    return obslist, actlist, rewlist


def final_eval_plot(outdir, env, obslist, actlist, rewlist, ncols=3):
    # obslist, actlist, and rewlist are list of lists
    # obslist_i, actlist_i, rewlist_i is list of obs, act, rew from rollout i
    
    # get relevant variables for plotting later
    horizon = get_horizon(env)
    nrollouts = len(obslist)
    nrows, rem = divmod(nrollouts, ncols)
    if rem > 0:  # add new row if rem > 0
        nrows += 1
        
    # convert obslist, actlist, rewlist to np.arrays for easier computation
    maxlen = 0
    for lst in obslist:
        maxlen = max(len(lst), maxlen)
    obsshape = obslist[0][0].shape[0]
    obslist_np = np.empty((nrollouts, maxlen, obsshape))
    obslist_np.fill(np.nan)
    actshape = actlist[0][0].shape[0]
    actlist_np = np.empty((nrollouts, maxlen, actshape))
    actlist_np.fill(np.nan)
    rewlist_np = np.empty((nrollouts, maxlen))  # rewlist contains scalers
    rewlist_np.fill(np.nan)
    
    for i, (olist, alist, rlist) in enumerate(zip(obslist, actlist, rewlist)):
        _len = len(olist)
        obslist_np[i, :_len] = olist
        actlist_np[i, :_len] = alist
        rewlist_np[i, :_len] = rlist

    # get plotting parameters so I don't need to do a str comparison each iteration
    if env == 'simglucose-adolescent2-v0':
        obs_extr = lambda obslist_i: obslist_i.flatten()
        act_extr = lambda actlist_i: actlist_i.flatten()
        
        ax1label = 'Glucose'
        ax1ylim = (0, 600)
        ax1xlabel = 'Steps (hours)'
        ax1ylabel = 'Glucose'
        plot_goal = lambda ax: ax.axhspan(70, 180, alpha=0.2, color='blue')
        
        ax2label = 'Basal'
        ax2ylim = (-5, 35)
        ax2ylabel = 'Basal'
        
        ax3label = 'Reward'
        ax3ylabel = 'Reward'
        
    elif env == 'MountainCarContinuous-v0':
        obs_extr = lambda obslist_i: obslist_i[:, 0].flatten()
        act_extr = lambda actlist_i: actlist_i.flatten()
        
        ax1label = 'X Position of Car'
        ax1ylim = (-1.5, 1)
        ax1xlabel = 'Steps'
        ax1ylabel = 'X Position of Car'
        plot_goal = lambda ax: ax.axhspan(0.45, 1, alpha=0.2, color='blue')
        
        ax2label = 'Force'
        ax2ylim = (-1.1, 1.1)
        ax2ylabel = 'Force'
        
        ax3label = 'Reward'
        ax3ylabel = 'Reward'
        
    elif env == 'Pendulum-v1':
        obs_extr = lambda obslist_i: np.arccos(obslist_i[:, 0]).flatten()
        act_extr = lambda actlist_i: actlist_i.flatten()
        
        ax1label = 'Angle'
        ax1ylim = (-3.3, 3.3)
        ax1xlabel = 'Steps'
        ax1ylabel = 'Angle (0 is upright position)'
        plot_goal = lambda ax: ax.axhline(y=0, alpha=0.2, linestyle='dotted', color='blue')
        
        ax2label = 'Torque'
        ax2ylim = (-2.2, 2.2)
        ax2ylabel = 'Torque'
        
        ax3label = 'Reward'
        ax3ylabel = 'Reward'
    
    
    print(f'Plotting observations, rewards, and actions over {nrollouts} rollout(s)...')
    
    # plot individual evaluations
    for k in range(nrollouts):
        olist = obs_extr(obslist_np[k])
        alist = act_extr(actlist_np[k])
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        ax1.plot(olist, color='blue', label=ax1label)
        ax1.set_ylim(ax1ylim)
        ax1.set_xlabel(ax1xlabel)
        ax1.set_ylabel(ax1ylabel, color='blue')
        plot_goal(ax1)

        # action axis
        ax2 = ax1.twinx()
        ax2.plot(alist, color='orange', alpha=0.4, label=ax2label)
        ax2.set_ylim(ax2ylim)
        ax2.set_ylabel(ax2ylabel, color='orange')
        
        # reward axis
        ax3 = ax1.twinx()
        ax3.plot(rewlist_np[k], color='red', label=ax3label)
        ax3.set_ylabel(ax3ylabel, color='red')
        
        # reposition action axis
        ax2.spines['right'].set_position(('outward', 60))
        
        fig.tight_layout()
        plotname = f'final_eval_{k+1:02d}.png'
        plotdir = osp.join(outdir, plotname)
        plt.savefig(plotdir)
        plt.close()
        
    
    # plot all on a single grid
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*ncols, 6*nrows))

    for k in range(nrollouts):
        olist = obs_extr(obslist_np[k])
        alist = act_extr(actlist_np[k])
        
        i, j = (0, 0) if k == 0 else divmod(k, ncols)
        
        ax1 = axs[i, j] if nrows > 1 else axs[j]
        ax1.plot(olist, color='blue', label=ax1label)
        ax1.set_ylim(ax1ylim)
        ax1.set_xlabel(ax1xlabel)
        ax1.set_ylabel(ax1ylabel, color='blue')
        plot_goal(ax1)

        # action axis
        
        ax2 = ax1.twinx()
        ax2.plot(alist, color='orange', alpha=0.4, label=ax2label)
        ax2.set_ylim(ax2ylim)
        ax2.set_ylabel(ax2ylabel, color='orange')
        
        # reward axis
        ax3 = ax1.twinx()
        ax3.plot(rewlist_np[k], color='red', label=ax3label)
        ax3.set_ylabel(ax3ylabel, color='red')
        
        # reposition action axis
        ax2.spines['right'].set_position(('outward', 60))

    fig.tight_layout()
    plotname = 'final_eval_all.png'
    plotdir = osp.join(outdir, plotname)
    plt.savefig(plotdir)
    plt.close()


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
    
    # don't touch these!
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
    
    register_simglucose(custom_rew=args.custom_reward)
    
    env = gym.make(args.env)
    env.np_random = np.random.default_rng(seed=args.seed)
    
    eval_env = gym.make(args.env)  # for evalutions during training
    eval_env.np_random = np.random.default_rng(seed=args.seed+1)
    
    final_env = gym.make(args.env)  # for final evalution after training
    final_env.np_random = np.random.default_rng(seed=args.seed+2)

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
        'net_arch': [256, 256]
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
        
        print(f'Doing {args.n_final_eval} rollouts...')
        obslist, actlist, rewlist = do_rollouts(model, args.alg, final_env, args.n_final_eval)


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

        print(f'Doing {args.n_final_eval} rollouts...')
        obslist, actlist, rewlist = do_rollouts(model, args.alg, final_env, args.n_final_eval)

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

        agent = NAF_Agent(state_size=state_size,
                          state_space=state_space,
                          action_size=action_size,
                          action_space=action_space,
                          horizon=horizon,
                          device=device, 
                          args=args,
                          writer=writer)
        
        print(f'model.device = {agent.device}')
        print(agent.qnetwork_local)
        
        if args.only_final_eval:
            # only do final evaluation
            print('Only doing final evaluation...')
            print('Loading model...')
            agent.load_model(modelname)
            
        else:
            # do training
            print('Starting training...')
            t_start = time()
            naf_runner(args, agent, env, eval_env, outdir, writer)
            t_end = time()
            print_timedelta(t_start, t_end, 'Training')
            print('Saving model...')
            torch.save(agent.qnetwork_local.state_dict(), modelname + '_.pth')
            # save parameter
            with open('runs/'+exp_name+".json", 'w') as f:
                json.dump(args.__dict__, f, indent=2)
                
        print(f'Doing {args.n_final_eval} rollouts...')
        obslist, actlist, rewlist = naf_final_eval(args, agent, final_env, writer)
    
    # make plots
    if not args.only_final_eval:
        training_eval_plots(outdir, args.n_eval_episodes)

    final_eval_plot(outdir, args.env, obslist, actlist, rewlist)
