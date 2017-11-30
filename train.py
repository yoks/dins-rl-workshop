#!/usr/bin/env python
import sys

from absl import app
from absl import flags

from agents import *
import settings
from baselines.logger import Logger, TensorBoardOutputFormat

import ppaquette_gym_super_mario

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ppaquette/SuperMarioBros-1-1-v0", "RL environment to train.")
flags.DEFINE_string("agent", "a2c", "RL algorithm to use.")
flags.DEFINE_integer("episodes", 10000, "Number of episodes")
flags.DEFINE_float("timesteps", 2000000, "Number of timesteps")



def _main(unused_argv):
    FLAGS(sys.argv)

    Logger.DEFAULT \
        = Logger.CURRENT \
        = Logger(dir=settings.ROOT_DIR + '/model',
                 output_formats=[TensorBoardOutputFormat('tensorboard')])

    if FLAGS.agent == "random":
        run_random_agent(FLAGS.env, FLAGS.episodes)
    elif FLAGS.agent == "dqn":
        train_dqn_agent(FLAGS.env, FLAGS.episodes)
    elif FLAGS.agent == "a2c":
        train_a2c_agent(FLAGS.env, FLAGS.timesteps)
    elif FLAGS.agent == "acktr":
        train_acktr_agent(FLAGS.env, FLAGS.timesteps)
    elif FLAGS.agent == "ppo":
        train_ppo(FLAGS.env, FLAGS.timesteps)


if __name__ == '__main__':
    app.run(_main)
