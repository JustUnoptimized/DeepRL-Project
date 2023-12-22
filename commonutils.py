from datetime import datetime
import gymnasium as gym
from simglucose.simulation.scenario_gen import RandomScenario
from bgrisks import simple_reward
from bgrisks import my_reward_closure


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


# import numpy as np
# from bgrisks import simple_reward
# from bgrisks import my_reward_closure
# from simglucose.simulation.env import risk_diff


# def pendulum_reward(x, x_next, u, t):
#     theta = np.arccos(x_next[0])
#     theta_prev = np.arccos(x[0])
#     theta_dt = theta - theta_prev
#     torque = u[0]

#     t1 = theta ** 2
#     t2 = 0.1 * (theta_dt ** 2)
#     t3 = 0.001 * (torque ** 2)

#     done = t == 200

#     return -(t1 + t2 + t3), done


# def mountaincar_reward(x_next, u, t):
#     r = -0.1 * (u[0] ** 2)
#     done = False
#     if x_next[0] >= 0.45:
#         r += 100
#         done = True

#     # t + 1 because t is 0-indexed
#     if t + 1 == 999:
#         done = True

#     return r, done


# def simglucose_reward(custom_reward, x, x_next):
#     if custom_reward == 'default':
#         r = risk_diff([x, x_next])
#     elif custom_reward == 'simple':
#         r = simple_reward([x_next])
#     else:
#         # magni or kovatchev reward
#         r = my_reward_closure(custom_reward)([x_next])

#     done = x_next[0] < 10 or x_next[0] > 600

#     return r, done


# def reward_fn(envname, custom_reward, x, x_next, u, t):
#     if envname == 'Pendulum-v1':
#         r, done = pendulum_reward(x, x_next, u, t)
#     elif envname == 'MountainCarContinuous-v0':
#         r, done = mountaincar_reward(x_next, u, t)
#     elif envname == 'simglucose-adolescent2-v0':
#         r, done = simglucose_reward(custom_reward, x, x_next)

#     return r, done