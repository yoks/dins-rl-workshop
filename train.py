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
flags.DEFINE_integer("timesteps", 2000000, "Steps to train")
flags.DEFINE_float("exploration_fraction", 0.5, "Exploration Fraction")
flags.DEFINE_boolean("prioritized", False, "prioritized_replay")
flags.DEFINE_boolean("dueling", False, "dueling")
flags.DEFINE_integer("num_cpu", 4, "number of cpus")
flags.DEFINE_float("lr", 5e-4, "Learning rate")

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")


def _main(unused_argv):
    FLAGS(sys.argv)

    if FLAGS.agent == "random":
        run_random_agent(FLAGS.env, FLAGS.timesteps)


if __name__ == '__main__':
    app.run(_main)
