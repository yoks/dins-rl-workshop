import gym

from wrappers import ToDiscreteWrapper, ProcessFrame84

import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds, explained_variance
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.policies import LstmPolicy
from baselines.a2c.a2c import Model, Runner
import tensorflow as tf
import time


def learn(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
    #model.load('./model/a2c/model.h5')
    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
            model.save('./model/a2c/model.h5')
    env.close()


def run(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes)
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    model.load('./model/a2c/model.h5')
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
    while True:
        runner.run()


def make_env(rank, env_id, seed):
    def _thunk():
        env = gym.make(env_id)
        env = ToDiscreteWrapper(env)
        env = ProcessFrame84(env)
        env.seed(seed + rank)
        env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
        gym.logger.setLevel(logging.WARN)
        return env
    return _thunk


def train_a2c_agent(env_id, timesteps, seed=0, num_cpu=2):
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i, env_id, seed) for i in range(num_cpu)])
    learn(LstmPolicy, env, seed, nsteps=20, nstack=4, total_timesteps=int(timesteps), vf_coef=0.5, ent_coef=0.01,
          max_grad_norm=0.5, lr=7e-4, lrschedule='constant', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100)


def run_a2c_agent(env_id, seed):
    env = gym.make(env_id)
    env = ToDiscreteWrapper(env)
    env = ProcessFrame84(env)
    env.seed(seed)
    env = SubprocVecEnv([make_env(0, env_id, seed)])
    run(LstmPolicy, env, seed, nsteps=5, nstack=4, total_timesteps=int(5000), vf_coef=0.5, ent_coef=0.01,
        max_grad_norm=0.5, lr=7e-4, lrschedule='constant', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100)


