import os
import gym
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.results_plotter import load_results, ts2xy
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from IPython.display import clear_output

from optical_rl_gym.envs.deeprmsa_env import DeepRMSAEnv

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

k_paths = 5
with open(f'/home/zceeas0/optical-rl-gym/examples/new_topologies/NSFNET_5-paths_CLSE.h5', 'rb') as f:
    topology = pickle.load(f)

# node probabilities from https://github.com/xiaoliangchenUCD/DeepRMSA/blob/6708e9a023df1ec05bfdc77804b6829e33cacfe4/Deep_RMSA_A3C.py#L77
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505, #add comma again
        0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
        0.07607608, 0.12012012, 0.01901902, 0.16916917])

loads = [50, 150, 300]
for load in loads: 
    # define the environment 
    env = DeepRMSAEnv(topology=topology, seed=10,
                    allow_rejection=False, # the agent cannot proactively reject a request
                    j=1, # consider only the first suitable spectrum block for the spectrum assignment
                    mean_service_holding_time=load/10, # value is not set as in the paper to achieve comparable reward values
                    episode_length=50, node_request_probabilities=node_request_probabilities)
    obs = env.reset()

    # define number of episodes(n_steps) and the length of each episode (episode)
    n_steps = 2000
    episode = 50 

    # initialize variables that store the rewards and episode service blocking rate
    rewards = []
    x = []
    blocking_rate = []

    for _ in range(n_steps):
        ep_reward = 0
        for _ in range(episode):
            action = env.action_space.sample() #choose a random action
            obs, reward, done, info = env.step(action)
            ep_reward = ep_reward + reward
        rewards.append(ep_reward)
        blocking_rate.append(info['episode_service_blocking_rate'])

    for t in range(n_steps):
        x.append(t)

    plt.plot(x, rewards)
    plt.ylim(-50,50)
    plt.xlabel("Number of episodes simulated")
    plt.ylabel("Episode Reward")
    plt.title("Random policy - {} Erlangs".format(load))
    plt.show()

    plt.plot(x,blocking_rate)
    plt.ylim(0,1)
    plt.xlabel("Number of episodes simulated")
    plt.ylabel("Episode service blocking rate")
    plt.title("Random policy - {} Erlangs".format(load))
    plt.show()
