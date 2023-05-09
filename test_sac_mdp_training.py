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
import random
from config import cfg_seed, cfg_env
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
args = parser.parse_args()


seed = cfg_seed
env_str = cfg_env
if cfg_env == "pendulum":
    env_name = "CustomPendulum-v0"
if cfg_env == "cartpole":
    env_name = "CustomCartPole-v0"

env = gym.make(env_name)

random(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)
torch.manual_seed(seed)


# Agent
agent = SAC(env.observation_space.shape[0], env.action_space)


# Memory
memory = ReplayMemory(args.replay_size, seed)

# Training Loop
total_numsteps = 0
updates = 0

train_epirew_list = []
train_steps_list = []
test_epirew_list = []
test_steps_list = []
for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    if total_numsteps > args.num_steps:
        break

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    train_epirew_list.append(episode_reward)
    train_steps_list.append(total_numsteps)

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward


                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        agent.save_checkpoint(env_name="custom_"+env_str)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")
        test_epirew_list.append(avg_reward)
        test_steps_list.append(total_numsteps)
        plt.plot(train_steps_list, train_epirew_list, label="train")
        plt.plot(test_steps_list, test_epirew_list, label="test")
        plt.legend()
        plt.savefig("fig_policy_optimization_curve_standardvae.png")
        plt.close()

env.close()
