import cv2
import gym
import numpy as np
from gym import spaces


# https://github.com/chris-chris/mario-rl-tutorial/blob/master/wrappers.py
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))
        self.prev_image = None

    def _observation(self, obs):
        return self.process(obs)

    def process(self, img):
        x_t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x_t = cv2.resize(x_t, (84, 84), interpolation=cv2.INTER_AREA)
        self.prev_image = x_t if self.prev_image is None else self.prev_image
        diff = cv2.absdiff(x_t, self.prev_image)
        self.prev_image = x_t
        diff = np.reshape(diff, (84, 84, 1))
        diff = np.nan_to_num(diff)
        return diff.astype(np.uint8)


class MarioActionSpaceWrapper(gym.ActionWrapper):
    """
      Wrapper to convert MultiDiscrete action space to Discrete

      Only supports one config, which maps to the most logical discrete space possible
    """
    mapping = {
        0: [0, 0, 0, 0, 0, 0],  # NOOP
        1: [1, 0, 0, 0, 0, 0],  # Up
        2: [0, 0, 1, 0, 0, 0],  # Down
        3: [0, 1, 0, 0, 0, 0],  # Left
        4: [0, 1, 0, 0, 1, 0],  # Left + A
        5: [0, 1, 0, 0, 0, 1],  # Left + B
        6: [0, 1, 0, 0, 1, 1],  # Left + A + B
        7: [0, 0, 0, 1, 0, 0],  # Right
        8: [0, 0, 0, 1, 1, 0],  # Right + A
        9: [0, 0, 0, 1, 0, 1],  # Right + B
        10: [0, 0, 0, 1, 1, 1],  # Right + A + B
        11: [0, 0, 0, 0, 1, 0],  # A
        12: [0, 0, 0, 0, 0, 1],  # B
        13: [0, 0, 0, 0, 1, 1],  # A + B
    }

    def __init__(self, env):
        super(MarioActionSpaceWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(14)

    def _action(self, action):
        return self.mapping.get(action)

    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if self.mapping[k] == action:
                return self.mapping[k]
        return 0
