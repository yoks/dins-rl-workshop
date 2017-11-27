#!/usr/bin/env python
import sys

from absl import app
from absl import flags

from agents import *

import ppaquette_gym_super_mario

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("agent", "dqn", "RL algorithm to use.")


def _main(unused_argv):
    FLAGS(sys.argv)

    if FLAGS.agent == "random":
        run_random_agent(FLAGS.env, FLAGS.episodes)
    elif FLAGS.agent == "dqn":
        run_dqn_agent(FLAGS.env)


if __name__ == '__main__':
    app.run(_main)
