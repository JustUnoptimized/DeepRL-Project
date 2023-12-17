import torch
import numpy as np
import json 
import random

from collections import deque
import time
import os.path as osp
# import gym
import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility
import pybullet_envs

import argparse
#import wandb
from torch.utils.tensorboard import SummaryWriter
from naf.agent import NAF_Agent
import matplotlib.pyplot as plt

from simglucose.simulation.scenario_gen import RandomScenario
from datetime import datetime


############################################
######          DELETE LATER          ######
import warnings
warnings.filterwarnings("ignore")
############################################


########### OBSOLETE--JUST USE RUNNER.PY ###########
def register_simglucose():
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())
    meal_scenario = RandomScenario(start_time=start_time, seed=1)

    gym.register(
        id='simglucose-adolescent2-v0',
        entry_point='simglucose.envs:T1DSimGymnaisumEnv',
        kwargs={'patient_name': 'adolescent#002',
                'custom_scenario': meal_scenario}
    )


########### OBSOLETE--JUST USE RUNNER.PY ###########
def make_plots(env, alg, obslist, actlist, rewlist):
    if env == 'simglucose-adolescent2-v0':
        # plot obs, obs safe zone, rew, action on single plot
        print('Plotting observations and actions over rollout...')
        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(obslist, color='blue', label='Glucose')
        ax1.set_ylim((0, 600))
        ax1.set_xlabel('Steps (hours)')
        ax1.set_ylabel('Glucose', color='blue')
        ax1.fill_between(np.arange(len(obslist)), 70, 180, alpha=0.2)

        # action axis
        ax2 = ax1.twinx()
        ax2.plot(actlist, color='orange', alpha=0.4, label='Basal')
        ax2.set_ylim((-5, 35))
        ax2.set_ylabel('Basal', color='orange')
        
        # reward axis
        ax3 = ax1.twinx()
        ax3.plot(rewlist, color='red', label='Reward')
        ax3.set_ylabel('Reward', color='red')
        
        # reposition action axis
        ax2.spines['right'].set_position(('outward', 60))
        
        fig.tight_layout()
        plotname = f'plots/{alg}_{env}.png'
        plt.savefig(plotname)
        plt.close()
        
    elif env == 'MountainCarContinuous-v0':
        print('Plotting Actions and Rewards')
        fig, ax1 = plt.subplots(figsize=(8, 6))
        # start rewlist from 1 because do_rollout appends nan at position 0
        ax1.plot(rewlist[1:], color='red', label='Reward')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward', color='red')

        # action axis
        ax2 = ax1.twinx()
        # start actlist from 1 because do_rollout appends nan at position 0
        ax2.plot(actlist[1:], color='orange', alpha=0.4, label='Action')
        ax2.set_ylim((-1.25, 1.25))
        ax2.set_ylabel('Force', color='orange')

        fig.tight_layout()
        plotname = f'plots/{alg}_{env}.png'
        plt.savefig(plotname)
        plt.close()
    elif env == 'Pendulum-v1':
        print('Plotting Actions and Rewards')
        fig, ax1 = plt.subplots(figsize=(8, 6))
        # start rewlist from 1 because do_rollout appends nan at position 0
        ax1.plot(rewlist[1:], color='red', label='Reward')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward', color='red')

        # action axis
        ax2 = ax1.twinx()
        # start actlist from 1 because do_rollout appends nan at position 0
        ax2.plot(actlist[1:], color='orange', alpha=0.4, label='Action')
        ax2.set_ylim((-2.25, 2.25))
        ax2.set_ylabel('Torque', color='orange')

        fig.tight_layout()
        plotname = f'plots/{alg}_{env}.png'
        plt.savefig(plotname)
        plt.close()


# very jank with a conditional return statement. Future me beware!
def evaluate(frame, agent, env, eval_runs, eval_scores, eval_ep_lens, writer, return_lists=False):
    ### has side effect of appending stuff to eval_scores and eval_ep_lens!!
    
    # holds total rollout reward over eval_runs trials
    # shape = len(timesteps) x eval_runs
    
    trial_scores = []
    trial_ep_lens = []
    
    # return a list of lists
    if return_lists:
        obslist = []
        actlist = []
        rewlist = []

    with torch.no_grad():
        for i in range(eval_runs):
            state, _ = env.reset()    
            score = 0  # total reward over rollout
            done = 0
            ep_len = 0
            if return_lists:
                obslist_i = [state]
                # this funky thing is just so all the elements of actlist have the same shape
                # which is equal to env.action_space.shape -- I am assuming this is a 1D box
                actlist_i = [np.array([np.nan for _ in range(env.action_space.shape[0])])]
                rewlist_i = [np.nan]
            while not done:
                action = agent.act_without_noise(state)
                state, reward, done, _ = env.step(action)
                score += reward
                ep_len += 1
                if return_lists:
                    obslist_i.append(state)
                    actlist_i.append(action)
                    rewlist_i.append(reward)
                if done:
                    trial_scores.append(score)
                    trial_ep_lens.append(ep_len)
                    if return_lists:
                        print(f'Rollout {i+1:02d} was {len(obslist_i)-1} actions long...')
                    break
            if return_lists:
                obslist.append(obslist_i)
                actlist.append(actlist_i)
                rewlist.append(rewlist_i)

    #wandb.log({"Reward": np.mean(scores), "Step": frame})
    writer.add_scalar("Reward", np.mean(trial_scores), frame)
    eval_scores.append(trial_scores)
    eval_ep_lens.append(trial_ep_lens)
    if return_lists:
        return obslist, actlist, rewlist


########### OBSOLETE--JUST USE RUNNER.PY ###########
def timer(start,end):
    """ Helper to print training time """
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTraining Time:  {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


########### OBSOLETE--JUST USE RUNNER.PY ###########
def run(args):
    """"NAF.
    
    Params
    ======

    """
    frames = args.frames
    eval_every = args.eval_every
    eval_runs = args.eval_runs
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    i_episode = 0
    state, _ = env.reset()
    score = 0 
    evaluate(0, test_env, eval_runs)
    for frame in range(1, frames+1):
        action = agent.act(state)

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done, t)
        

        state = next_state
        score += reward

        if frame % eval_every == 0:
            evaluate(frame, test_env, eval_runs)

        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tFrame [{}/{}] \tAverage Score: {:.2f}'.format(i_episode, frame, frames, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame [{}/{}] \tAverage Score: {:.2f}'.format(i_episode,frame, frames, np.mean(scores_window)))
            i_episode +=1 
            state, _ = env.reset()
            score = 0
    obslist, actlist, rewlist = evaluate(frame, final_env, 1, return_lists=True)
    return obslist, actlist, rewlist


def naf_runner(args, agent, env, eval_env, outdir, writer):
    # adapted run() so I can call it from runner.py    
    frames = args.frames
    eval_every = args.eval_freq
    eval_runs = args.n_eval_episodes
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    i_episode = 0
    
    state, _ = env.reset()
    ep = 1
    t = 0
    score = 0
    
    # holds timesteps of when evaluations happened
    timesteps = []
    # holds total rollout reward over eval_runs trials
    # shape = len(timesteps) x eval_runs
    eval_scores = []
    # holds rollout lengths over eval_runs trials
    # shape = len(timesteps) x eval_runs
    eval_ep_lens = []
    

    # use dummy lists for eval_scores and eval_ep_lens because
    # we want to start averaging after training starts
    evaluate(0, agent, eval_env, eval_runs, [], [], writer, return_lists=False)
    for frame in range(1, frames+1):
        action = agent.act(state, t)

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done, t, ep)

        state = next_state
        t += 1
        score += reward

        if frame % eval_every == 0:
            # has side effect of appending stuff to eval_scores and eval_ep_lens
            evaluate(frame, agent, eval_env, eval_runs, eval_scores, eval_ep_lens, writer, return_lists=False)
            timesteps.append(frame)

        if done:
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            print('\rEpisode {}\tFrame [{}/{}] \tAverage Score: {:.2f}'.format(i_episode, frame, frames, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame [{}/{}] \tAverage Score: {:.2f}'.format(i_episode,frame, frames, np.mean(scores_window)))
            i_episode +=1 
            
            ep += 1
            t = 0
            score = 0
            # fit local linear dynamics and do ilqg
            agent.maybe_flld_ilqg()
            # toggle between naf policy and ilqg policy with prob {p, 1-p}, defined in args
            agent.maybe_toggle_policy()
            
            state, _ = env.reset()
            
    # store training evaluation info for plotting later
    timesteps = np.array(timesteps)
    eval_scores = np.array(eval_scores)
    eval_ep_lens = np.array(eval_ep_lens)
    
    # sanity check that everything is as expected
    assert eval_scores.shape == eval_ep_lens.shape
    assert eval_scores.shape[0] == timesteps.shape[0]
    assert eval_scores.shape[1] == eval_runs
    
    outfile = osp.join(outdir, 'evaluations.npz')
    # arrays saved are kept consistent with those used in stable baselines 3
    np.savez(outfile, timesteps=timesteps, results=eval_scores, ep_lengths=eval_ep_lens)


def naf_final_eval(args, agent, final_env, writer): 
    frames = args.frames
    eval_runs = args.n_final_eval
    
    obslist, actlist, rewlist = evaluate(frames, agent, final_env, eval_runs, [], [], writer, return_lists=True)
    return obslist, actlist, rewlist


########### OBSOLETE--JUST USE RUNNER.PY ###########
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-info", type=str, default="Experiment-1",
                     help="Name of the Experiment (default: Experiment-1)")
    parser.add_argument('-env', type=str, default="Pendulum-v1",
                     help='Name of the environment (default: Pendulum-v1)')
    parser.add_argument('-f', "--frames", type=int, default=40000,
                     help='Number of training frames (default: 40000)')    
    parser.add_argument("--eval_every", type=int, default=5000,
                     help="Evaluate the current policy every X steps (default: 5000)")
    parser.add_argument("--eval_runs", type=int, default=2,
                     help="Number of evaluation runs to evaluate - averating the evaluation Performance over all runs (default: 3)")
    parser.add_argument('-mem', type=int, default=100000,
                     help='Replay buffer size (default: 100000)')
    parser.add_argument('-per', type=int, choices=[0,1],  default=0,
                     help='Use prioritized experience replay (default: False)')
    parser.add_argument('-b', "--batch_size", type=int, default=256,
                     help='Batch size (default: 128)')
    parser.add_argument('-nstep', type=int, default=1,
                     help='nstep_bootstrapping (default: 1)')
    parser.add_argument("-d2rl", type=int, choices=[0,1], default=0,
                     help="Using D2RL Deep Dense NN Architecture if set to 1 (default: 0)")
    parser.add_argument('-l', "--layer_size", type=int, default=256,
                     help='Neural Network layer size (default: 256)')
    parser.add_argument('-g', "--gamma", type=float, default=0.99,
                     help='Discount factor gamma (default: 0.99)')
    parser.add_argument('-t', "--tau", type=float, default=0.005,
                     help='Soft update factor tau (default: 0.005)')
    parser.add_argument('-lr', "--learning_rate", type=float, default=1e-3,
                     help='Learning rate (default: 1e-3)')
    parser.add_argument('-u', "--update_every", type=int, default=1,
                     help='update the network every x step (default: 1)')
    parser.add_argument('-n_up', "--n_updates", type=int, default=1,
                     help='update the network for x steps (default: 1)')
    parser.add_argument('-s', "--seed", type=int, default=0,
                     help='random seed (default: 0)')
    parser.add_argument("--clip_grad", type=float, default=1.0, help="Clip gradients (default: 1.0)")
    parser.add_argument("--loss", type=str, choices=["mse", "huber"], default="mse", help="Choose loss type MSE or Huber loss (default: mse)")
    
    args = parser.parse_args()
    
    for k, v in vars(args).items():
        print(f'{k:<20s} : {v}')
        
    #wandb.init(project="naf", name=args.info)
    #wandb.config.update(args)
    writer = SummaryWriter("runs/"+args.info)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    register_simglucose()
    
    env = gym.make(args.env) #CartPoleConti
    test_env = gym.make(args.env)
    final_env = gym.make(args.env)

    seed = args.seed
    np.random.seed(seed)
    env.np_random = np.random.default_rng(seed=seed)
    test_env.np_random = np.random.default_rng(seed=seed+1)
    final_env.np_random = np.random.default_rng(seed=1234)
    
    action_size = env.action_space.shape[0]
    state_size = env.observation_space.shape[0]
    action_space = env.action_space

    agent = NAF_Agent(state_size=state_size,
                      action_size=action_size,
                      action_space=action_space,
                      device=device, 
                      args= args,
                      writer=writer)

    t0 = time.time()
    obslist, actlist, rewlist = run(args)
    t1 = time.time()
    
    timer(t0, t1)
    torch.save(agent.qnetwork_local.state_dict(), "NAF_"+args.info+"_.pth")
    # save parameter
    with open('runs/'+args.info+".json", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    make_plots(args.env, 'naf', obslist, actlist, rewlist)
