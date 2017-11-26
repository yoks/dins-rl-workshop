#!/usr/bin/env python
import sys

from absl import app
from absl import flags

from agents import *

import datetime
import ppaquette_gym_super_mario

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("agent", "random", "RL algorithm to use.")
flags.DEFINE_integer("episodes", 10000, "Number of episodes")

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")


def _main(unused_argv):
    FLAGS(sys.argv)

    if FLAGS.agent == "random":
        run_random_agent(FLAGS.env, FLAGS.episodes)


if __name__ == '__main__':
    app.run(_main)
