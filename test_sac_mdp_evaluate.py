import argparse
import datetime
import gym
import custom_gym
import numpy as np
import itertools
import torch
from sac import SAC
# from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
# parser.add_argument('--env-name', default="CustomPendulum-v0",
parser.add_argument('--env-name', default="CustomCartPole-v0",
                    help='custom')
# parser.add_argument('--policy', default="Gaussian",
#                     help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
# parser.add_argument('--eval', type=bool, default=True,
#                     help='Evaluates a policy a policy every 10 episode (default: True)')
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
# parser.add_argument('--batch_size', type=int, default=256, metavar='N',
#                     help='batch size (default: 256)')
# parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
#                     help='maximum number of steps (default: 1000000)')
# parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
#                     help='hidden size (default: 256)')
# parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
#                     help='model updates per simulator step (default: 1)')
# parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
#                     help='Steps sampling random actions (default: 10000)')
# parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
#                     help='Value target update per no. of updates per step (default: 1)')
# parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
#                     help='size of replay buffer (default: 10000000)')
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

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space)

if args.env_name=="CustomPendulum-v0":
    agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_pendulum_", evaluate=False)
if args.env_name=="CustomCartPole-v0":
    agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_cartpole_", evaluate=False)
#Tesnorboard
# writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Training Loop
total_numsteps = 0
updates = 0


state = env.reset()
done = False
state_history = []

# env.env.env.m = 10.

print("env.env.env.get_params()",env.env.env.get_params())
# env.env.env.set_params(c=0.3)
print("env.env.env.get_params()",env.env.env.get_params())

total_reward = 0
while not done:
    state_history.append(state)
    action = agent.select_action(state)  # Sample action from policy
    next_state, reward, done, _ = env.step(action) # Step
    total_reward += reward
    state = next_state

env.close()

state_history = np.array(state_history)
print("total_reward",total_reward)
import matplotlib.pyplot as plt
if args.env_name=="CustomPendulum-v0":
    plt.plot(state_history[:,0], state_history[:,1])
    plt.plot(state_history[0,0], state_history[0,1], "o")
if args.env_name=="CustomCartPole-v0":
    # plt.plot(state_history[:,0], state_history[:,2])
    # plt.plot(state_history[0,0], state_history[0,2], "o")
    plt.plot(state_history[:,2], state_history[:,3])
    plt.plot(state_history[0,2], state_history[0,3], "o")
plt.show()
