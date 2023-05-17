import argparse
import datetime
import gym
import custom_gym
import numpy as np
import itertools
import torch
from sac import SAC
from utils import torch_from_numpy
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory
import random
from config import cfg_seed, cfg_env, cfg_z_dim

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
# parser.add_argument('--cuda', action="store_true",
#                     help='run on CUDA (default: False)')
args = parser.parse_args()


seed = cfg_seed
env_str = cfg_env
if cfg_env == "pendulum":
    env_name = "CustomPendulum-v0"
if cfg_env == "cartpole":
    env_name = "CustomCartPole-v0"

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(env_name)

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)
env.action_space.seed(seed)


import vi_base
import vi_iw
import pickle
s_dim = env.reset().flatten().shape[0]
a_dim = env.action_space.sample().flatten().shape[0]
z_dim = cfg_z_dim
offline_data = pickle.load(open("offline_data_"+env_str+".pkl","rb"))
debug_info = pickle.load(open("offline_data_debug_info_"+env_str+".pkl","rb"))
debug_info = np.array(debug_info)
args_init_dict = {"offline_data": offline_data,
             "s_dim": s_dim,
             "a_dim": a_dim,
             "z_dim": z_dim,
             "mdp_policy":None,
             "bamdp_policy":None,
             "debug_info": None,#debug_info,
             "env" : env,
             "ckpt_suffix" : env_str,}


# Agent
agent = SAC(env.observation_space.shape[0]+z_dim*2, env.action_space)



if 1:
    agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_"+env_str+"_bamdp_standardvae_", evaluate=True)
    vi = vi_base.baseVI(args_init_dict)
    vi.load(ckpt_key="unweighted")


if 0:
    agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_"+env_str+"_bamdp_weightedvae_", evaluate=True)
    # agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_pendulum_bamdp_realbamdpdebug_", evaluate=True)
    vi = vi_iw.iwVI(args_init_dict)
    


# Memory
# memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

avg_reward = 0.
episodes = 5
for _  in range(episodes):
    state = env.reset(fix_init=True)
    belief = vi.get_belief()
    episode_reward = 0
    done = False
    sads_array=np.empty((0,s_dim*2+a_dim))
    belief_array=np.empty((0,z_dim*2))
    rew_list=[]
    while not done:
        aug_state = np.hstack([state, belief.numpy()])
        action = agent.select_action(aug_state, evaluate=True)

        next_state, reward, done, _ = env.step(action)
        sads_array = np.vstack([sads_array, 
                                np.hstack([state, action, next_state-state])])
        belief_array = np.vstack([belief_array,belief.numpy()])
        episode_reward += reward
        rew_list.append(reward)

        state = next_state
        belief = vi.get_belief(sads_array=sads_array)

        # debug plot nll per step
        if 0:
            plotnum=40
            losslist=[]
            zlist=[]
            for i in range(plotnum):
                zmin = min(vi.mulogvar_offlinedata[:,0].min(),belief[0])
                zmax = max(vi.mulogvar_offlinedata[:,0].max(),belief[0])
                z = zmin + (zmax-zmin)*( 1*(i/plotnum)-0)
                zlist.append(z)
                losslist.append(vi.get_nll(sads_array,z))
            plt.plot(zlist,losslist)
            plt.show()

    print("\nenv.get_params()",env.get_params())
    # debug plot per episode
    if 1:
        print("episode_reward",episode_reward)
        plt.plot(sads_array[0,0],sads_array[0,1],"o")
        plt.plot(sads_array[:,0],sads_array[:,1])
        plt.show()
        # plt.plot(belief_array[:,0])
        # plt.ylabel("belief_mu")
        # plt.xlabel("timestamp")
        # plt.show()
        # plt.plot(rew_list)
        # plt.ylabel("reward")
        # plt.xlabel("timestamp")
        # plt.show()
    avg_reward += episode_reward
avg_reward /= episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")

env.close()
