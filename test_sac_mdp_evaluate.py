import argparse
import datetime
import gym
import custom_gym
import numpy as np
import random
import itertools
import torch
from sac import SAC
from config import cfg_seed, cfg_env

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
args = parser.parse_args()


# Environment
seed = cfg_seed
env_str = cfg_env
if cfg_env == "pendulum":
    env_name = "CustomPendulum-v0"
if cfg_env == "cartpole":
    env_name = "CustomCartPole-v0"


env = gym.make(env_name)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)
env.action_space.seed(seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space)

agent.load_checkpoint(ckpt_path="checkpoints/sac_checkpoint_custom_"+env_str+"_", evaluate=False)

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
if env_name=="CustomPendulum-v0":
    plt.plot(state_history[:,0], state_history[:,1])
    plt.plot(state_history[0,0], state_history[0,1], "o")
    plt.show()
if env_name=="CustomCartPole-v0":
    plt.plot(state_history[:,0], state_history[:,1])
    plt.plot(state_history[0,0], state_history[0,1], "o")
    plt.show()
    plt.plot(state_history[:,2], state_history[:,3],"-x")
    plt.plot(state_history[0,2], state_history[0,3], "o")
    plt.show()
