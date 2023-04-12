import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CustomPendulum-v0',
    entry_point='custom_gym.envs:CustomPendulumEnv',
    #timestep_limit=200,
    max_episode_steps=200,
)

register(
    id='CustomCartPole-v0',
    entry_point='custom_gym.envs:CustomCartPoleEnv',
    #timestep_limit=200,
    max_episode_steps=200,
)
