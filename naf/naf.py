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


def evaluate(frame, agent, env, eval_runs, eval_scores, eval_ep_lens, writer):
    ### has side effect of appending stuff to eval_scores and eval_ep_lens!!
    
    trial_scores = []
    trial_ep_lens = []

    with torch.no_grad():
        for i in range(eval_runs):
            state, _ = env.reset()    
            score = 0  # total reward over rollout
            done = 0
            ep_len = 0
            while not done:
                action = agent.act_without_noise(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                ep_len += 1
                if done:
                    trial_scores.append(score)
                    trial_ep_lens.append(ep_len)
                    break

    #wandb.log({"Reward": np.mean(scores), "Step": frame})
    writer.add_scalar("Reward", np.mean(trial_scores), frame)
    eval_scores.append(trial_scores)
    eval_ep_lens.append(trial_ep_lens)


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
    

    evaluate(0, agent, eval_env, eval_runs, eval_scores, eval_ep_lens, writer)
    timesteps.append(0)
    for frame in range(1, frames+1):
        action = agent.act(state, t)

        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done, t, ep)

        state = next_state
        t += 1
        score += reward

        if frame % eval_every == 0:
            # has side effect of appending stuff to eval_scores and eval_ep_lens
            evaluate(frame, agent, eval_env, eval_runs, eval_scores, eval_ep_lens, writer)
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
