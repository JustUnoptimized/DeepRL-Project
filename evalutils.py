import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from commonutils import get_horizon


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
