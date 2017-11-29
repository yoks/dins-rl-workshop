import gym

from wrappers import MarioActionSpaceWrapper, ProcessFrame84

import os.path as osp
import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds, explained_variance
from baselines import bench
from baselines.acktr.acktr_disc import Model, Runner
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.acktr.policies import CnnPolicy
import tensorflow as tf
import time
import numpy as np

def learn(policy, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
          nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
          kfac_clip=0.001, save_interval=None, lrschedule='linear'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda : Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
    =nsteps, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                lrschedule=lrschedule)
    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
    nbatch = nenvs*nsteps
    tstart = time.time()
    coord = tf.train.Coordinator()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=coord, start=True)
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular("rewards", float(np.sum(rewards)))
        logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
    coord.request_stop()
    coord.join(enqueue_threads)
    env.close()


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


def train_acktr_agent(env_id, timesteps, seed=0, num_cpu=2):
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i, env_id, seed) for i in range(num_cpu)])
    learn(CnnPolicy, env, seed, total_timesteps=int(timesteps * 1.1), nprocs=num_cpu, save_interval=50)


