
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import math



class CustomPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=10
        self.max_torque=2.
        self.dt=.1
        # self.dt=.1
        self.viewer = None

        # high = np.array([1., 1., self.max_speed])
        high = np.array([4.*np.pi, self.max_speed])
        # self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(low=-np.float32(self.max_torque), high=np.float32(self.max_torque), shape=(1,))
        self.observation_space = spaces.Box(low=-np.float32(high), high=np.float32(high))

        # self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self,u):
        th, thdot = self.state # th := theta

        g = 10.
        # m = 1.
        m = self.m
        l = 1.
        c = self.c
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        #costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        rew = self.rew_fn(self._get_obs(), u)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*( -c*thdot + u)) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), rew, False, {}


    def rew_fn(self,o,u):

        #r = np.sqrt((o[0]**2)+(o[1]**2))
        #th = math.atan2(o[1]/r, o[0]/r)
        th = o[0]
        thdot = o[1]
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)
        # costs = th**2 + .01*thdot**2 + .001*(u**2) # bad even for real bamdp
        return -np.array([costs]).reshape(1)[0]


    def reset(self, fix_init=False):
        # high = np.array([np.pi/3, 1.])

        self.state = np.array([np.pi, 0.0])
        if not fix_init:
            high = np.array([0.8*np.pi, 0.5])
            self.state += self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        self.m = 0.5 + 0*0.*np.random.rand() # coeff * [0,1)

        # self.c = 0.25*np.random.rand() + 0.05# coeff * [0,1)
        self.c = 0.3*np.random.rand() + 0.0# coeff * [0,1)

        # self.env_param = np.array([self.m, self.c])
        # self.c=0
        print("c =",self.c)
        return self._get_obs().astype(np.float32)


    def get_params(self):
        return np.array([self.m, self.c])

    def set_params(self, m=None, c=None):
        if m is not None:
            self.m = m
        if c is not None:
            self.c = c


    def _get_obs(self):
        theta, thetadot = self.state
        # return np.array([np.cos(theta), np.sin(theta), thetadot])
        return np.array([theta, thetadot])


    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
