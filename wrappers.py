import gym
from gym import spaces
import numpy as np
import cv2


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3))
        self.window = cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    def _observation(self, obs):
        return self.process(obs)

    def process(self, img):
        x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        diff = np.reshape(x_t, (84, 84, 3))
        diff = np.nan_to_num(diff)
        #cv2.imshow('frame', diff)
        #cv2.waitKey(1)
        return diff.astype(np.uint8)


class ToDiscreteWrapper(gym.ActionWrapper):
    """
        Wrapper to convert MultiDiscrete action space to Discrete

        Only supports one config, which maps to the most logical discrete space possible
    """

    def __init__(self, env):
        super(ToDiscreteWrapper, self).__init__(env)
        self.mapping = {
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
        self.action_space = spaces.Discrete(14)

    def _action(self, action):
        return self.mapping.get(action)

    def _reverse_action(self, action):
        for k in self.mapping.keys():
            if self.mapping[k] == action:
                return self.mapping[k]
        return 0
