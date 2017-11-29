#!/usr/bin/env python
import sys

from absl import app
from absl import flags

import gym

import ppaquette_gym_super_mario

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("agent", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("episodes", 10000, "Number of episodes")
flags.DEFINE_float("timesteps", 2e10, "Number of episodes")


def _main(unused_argv):
    FLAGS(sys.argv)
    env = gym.make(FLAGS.env)
    env.reset()


if __name__ == '__main__':
    app.run(_main)
