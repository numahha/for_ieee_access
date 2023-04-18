import gym
import custom_gym
import numpy as np
import torch
import pickle


def getOfflineData(env_name, episode_num, seed):

    if env_name=='CustomPendulum-v0':
        env_suffix="pendulum"
    if env_name=='CustomCartPole-v0':
        env_suffix="cartpole"

    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    def random_rollout():
        transition_data = []
        o = env.reset()
        while True:
            a = env.action_space.sample()
            o_next, r, done, _ = env.step(a)
            transition_data.append(np.hstack([o,a,o_next,r,done]))
            #transition_data.append(np.hstack([o,a,o_next-o,r]))
            o = o_next
            if done:
                break
        transition_data = np.vstack(transition_data)
        return transition_data, env.env.env.get_params()


    offline_data = []
    debug_info = []
    for i in range(episode_num):
        transition_data, real_env_param = random_rollout()
        transition_data = torch.from_numpy(transition_data.astype(np.float32))
        offline_data.append(transition_data)
        debug_info.append(real_env_param)

    f = open('offline_data_'+env_suffix+'.pkl', 'wb')
    pickle.dump(offline_data, f)
    f.close()

    f2 = open('offline_data_debug_info_'+env_suffix+'.pkl', 'wb')
    pickle.dump(debug_info, f2)
    f2.close()
