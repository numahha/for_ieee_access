import argparse
import datetime
import gym
import custom_gym
import numpy as np
import itertools
import torch
import random
from sac import SAC
from utils import torch_from_numpy
import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="CustomPendulum-v0",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
# parser.add_argument('--policy', default="Gaussian",
#                     help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
# parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
#                     help='discount factor for reward (default: 0.99)')
# parser.add_argument('--tau', type=float, default=0.005, metavar='G',
#                     help='target smoothing coefficient(τ) (default: 0.005)')
# parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
#                     help='learning rate (default: 0.0003)')
# parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
#                     help='Temperature parameter α determines the relative importance of the entropy\
#                             term against the reward (default: 0.2)')
# parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
#                     help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
# parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
parser.add_argument('--num_steps', type=int, default=40001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
# parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
#                     help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
# parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
# parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
#                     help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
# parser.add_argument('--cuda', action="store_true",
#                     help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

import vi_base
import vi_iw
import pickle
s_dim = env.reset().flatten().shape[0]
a_dim = env.action_space.sample().flatten().shape[0]
z_dim = 1
offline_data = pickle.load(open("offline_data.pkl","rb"))
debug_info = pickle.load(open("offline_data_debug_info.pkl","rb"))
debug_info = np.array(debug_info)
args_init_dict = {"offline_data": offline_data,
             "s_dim": s_dim,
             "a_dim": a_dim,
             "z_dim": z_dim,
#              "policy":agent.select_action,
             "mdp_policy":None,
             "debug_info": None,#debug_info,
             "env" : env}


# Agent
agent = SAC(env.observation_space.shape[0]+z_dim*2, env.action_space)



# agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_pendulum_bamdp_standardvae_", evaluate=True)
# vi = vi_base.baseVI(args_init_dict)
agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_pendulum_bamdp_weightedvae_", evaluate=True)
vi = vi_iw.iwVI(args_init_dict)


vi.load()


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

avg_reward = 0.
episodes = 1
for _  in range(episodes):
    state = env.reset()
    belief = vi.get_belief()
    episode_reward = 0
    done = False
    sads_array=np.empty((0,s_dim*2+a_dim))
    belief_array=np.empty((0,z_dim*2))
    while not done:
        aug_state = np.hstack([state, belief.numpy()])
        action = agent.select_action(aug_state, evaluate=True)

        next_state, reward, done, _ = env.step(action)
        sads_array = np.vstack([sads_array, 
                                np.hstack([state, action, next_state-state])])
        belief_array = np.vstack([belief_array,belief.numpy()])
        episode_reward += reward

        state = next_state
        belief = vi.get_belief(sads_array=sads_array)

        # debug plot
        if 1:
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
    plt.plot(sads_array[0,0],sads_array[0,1],"o")
    plt.plot(sads_array[:,0],sads_array[:,1])
    plt.show()
    plt.plot(belief_array[:,0])
    plt.show()

    avg_reward += episode_reward
avg_reward /= episodes

print("----------------------------------------")
print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
print("----------------------------------------")

env.close()
