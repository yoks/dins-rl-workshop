#!/usr/bin/env python
import sys
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from wrappers import MarioActionSpaceWrapper, ProcessFrame84

import os, logging, gym


def make_env(rank, env_id, seed):
    def _thunk():
        env = gym.make(env_id)
        env = MarioActionSpaceWrapper(env)
        env = ProcessFrame84(env)
        env.seed(seed + rank)
        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
        gym.logger.setLevel(logging.WARN)
        return env
    return _thunk


def train_ppo(env_id, num_timesteps, seed=12):
    import gym
    import logging
    import multiprocessing
    import tensorflow as tf
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    nenvs = 4
    env = SubprocVecEnv([make_env(i, env_id, seed) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    ppo2.learn(policy=CnnPolicy, env=env, nsteps=32, nminibatches=4,
               lam=0.95, gamma=0.9, noptepochs=4, log_interval=1,
               ent_coef=.01,
               lr=lambda f: f * 3e-3,
               cliprange=lambda f: f * 0.1,
               total_timesteps=int(num_timesteps * 1.1), save_interval=10)