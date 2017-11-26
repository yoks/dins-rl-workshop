# https://github.com/chris-chris/mario-rl-tutorial Apache 2

import os.path as osp
import time
import os
import logging
import datetime

import baselines.common.tf_util as U
import gym
import joblib
import numpy as np
import settings
import tensorflow as tf
from baselines import bench
from baselines import logger
from baselines.acktr import kfac
from baselines.acktr.utils import Scheduler, find_trainable_variables
from baselines.acktr.utils import cat_entropy, mse
from baselines.acktr.utils import conv, fc, dense, conv_to_fc, sample, kl_div
from baselines.acktr.utils import discount_with_dones
from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.logger import Logger, TensorBoardOutputFormat


from wrappers import MarioActionSpaceWrapper, ProcessFrame84

max_mean_reward = 0
last_filename = ""

start_time = datetime.datetime.now().strftime("%Y%m%d%H%M")

logdir = "tensorboard/%s" % start_time

Logger.DEFAULT \
    = Logger.CURRENT \
    = Logger(dir=None,
             output_formats=[TensorBoardOutputFormat(logdir)])

class CnnPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv * nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc * nstack)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape)  # obs
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32) / 255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=32, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x: x)
            vf = fc(h4, 'v', 1, act=lambda x: x)

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = []  # not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X: ob})
            return a, v, []  # dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X: ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value


class GaussianMlpPolicy(object):
    def __init__(self, ob_dim, ac_dim):
        # Here we'll construct a bunch of expressions, which will be used in two places:
        # (1) When sampling actions
        # (2) When computing loss functions, for the policy update
        # Variables specific to (1) have the word "sampled" in them,
        # whereas variables specific to (2) have the word "old" in them
        ob_no = tf.placeholder(tf.float32, shape=[None, ob_dim * 2], name="ob")  # batch of observations
        oldac_na = tf.placeholder(tf.float32, shape=[None, ac_dim], name="ac")  # batch of actions previous actions
        oldac_dist = tf.placeholder(tf.float32, shape=[None, ac_dim * 2],
                                    name="oldac_dist")  # batch of actions previous action distributions
        adv_n = tf.placeholder(tf.float32, shape=[None], name="adv")  # advantage function estimate
        oldlogprob_n = tf.placeholder(tf.float32, shape=[None],
                                      name='oldlogprob')  # log probability of previous actions
        wd_dict = {}
        h1 = tf.nn.tanh(
            dense(ob_no, 64, "h1", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        h2 = tf.nn.tanh(
            dense(h1, 64, "h2", weight_init=U.normc_initializer(1.0), bias_init=0.0, weight_loss_dict=wd_dict))
        mean_na = dense(h2, ac_dim, "mean", weight_init=U.normc_initializer(0.1), bias_init=0.0,
                        weight_loss_dict=wd_dict)  # Mean control output
        self.wd_dict = wd_dict
        self.logstd_1a = logstd_1a = tf.get_variable("logstd", [ac_dim], tf.float32,
                                                     tf.zeros_initializer())  # Variance on outputs
        logstd_1a = tf.expand_dims(logstd_1a, 0)
        std_1a = tf.exp(logstd_1a)
        std_na = tf.tile(std_1a, [tf.shape(mean_na)[0], 1])
        ac_dist = tf.concat([tf.reshape(mean_na, [-1, ac_dim]), tf.reshape(std_na, [-1, ac_dim])], 1)
        sampled_ac_na = tf.random_normal(tf.shape(ac_dist[:, ac_dim:])) * ac_dist[:, ac_dim:] + ac_dist[:,
                                                                                                :ac_dim]  # This is the sampled action we'll perform.
        logprobsampled_n = - U.sum(tf.log(ac_dist[:, ac_dim:]), axis=1) - 0.5 * tf.log(
            2.0 * np.pi) * ac_dim - 0.5 * U.sum(
            tf.square(ac_dist[:, :ac_dim] - sampled_ac_na) / (tf.square(ac_dist[:, ac_dim:])),
            axis=1)  # Logprob of sampled action
        logprob_n = - U.sum(tf.log(ac_dist[:, ac_dim:]), axis=1) - 0.5 * tf.log(2.0 * np.pi) * ac_dim - 0.5 * U.sum(
            tf.square(ac_dist[:, :ac_dim] - oldac_na) / (tf.square(ac_dist[:, ac_dim:])),
            axis=1)  # Logprob of previous actions under CURRENT policy (whereas oldlogprob_n is under OLD policy)
        kl = U.mean(kl_div(oldac_dist, ac_dist, ac_dim))
        # kl = .5 * U.mean(tf.square(logprob_n - oldlogprob_n)) # Approximation of KL divergence between old policy used to generate actions, and new policy used to compute logprob_n
        surr = - U.mean(adv_n * logprob_n)  # Loss function that we'll differentiate to get the policy gradient
        surr_sampled = - U.mean(logprob_n)  # Sampled loss of the policy
        self._act = U.function([ob_no],
                               [sampled_ac_na, ac_dist, logprobsampled_n])  # Generate a new action and its logprob
        # self.compute_kl = U.function([ob_no, oldac_na, oldlogprob_n], kl) # Compute (approximate) KL divergence between old policy and new policy
        self.compute_kl = U.function([ob_no, oldac_dist], kl)
        self.update_info = (
        (ob_no, oldac_na, adv_n), surr, surr_sampled)  # Input and output variables needed for computing loss
        U.initialize()  # Initialize uninitialized TF variables

    def act(self, ob):
        ac, ac_dist, logp = self._act(ob[None])
        return ac[0], ac_dist[0], logp[0]


class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=32, nsteps=20,
                 nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                 kfac_clip=0.001, lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=nprocs,
                                inter_op_parallelism_threads=nprocs)
        config.gpu_options.allow_growth = True
        from tensorflow.python import debug as tf_debug

        self.sess = sess = tf.Session(config=config)

        nact = ac_space.n
        nbatch = nenvs * nsteps
        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        PG_LR = tf.placeholder(tf.float32, [])
        VF_LR = tf.placeholder(tf.float32, [])

        self.model = step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        self.model2 = train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_model.pi, labels=A)
        self.logits = logits = train_model.pi

        ##training loss
        pg_loss = tf.reduce_mean(ADV * logpac)
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        pg_loss = pg_loss - ent_coef * entropy
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        train_loss = pg_loss + vf_coef * vf_loss

        ##Fisher loss construction
        self.pg_fisher = pg_fisher_loss = -tf.reduce_mean(logpac)
        sample_net = train_model.vf + tf.random_normal(tf.shape(train_model.vf))
        self.vf_fisher = vf_fisher_loss = - vf_fisher_coef * tf.reduce_mean(
            tf.pow(train_model.vf - tf.stop_gradient(sample_net), 2))
        self.joint_fisher = joint_fisher_loss = pg_fisher_loss + vf_fisher_loss

        self.params = params = find_trainable_variables("model")

        self.grads_check = grads = tf.gradients(train_loss, params)

        with tf.device('/gpu:0'):
            self.optim = optim = kfac.KfacOptimizer(learning_rate=PG_LR, clip_kl=kfac_clip, \
                                                    momentum=0.9, kfac_update=1, epsilon=0.01, \
                                                    stats_decay=0.99, async=1, cold_iter=10,
                                                    max_grad_norm=max_grad_norm)

            update_stats_op = optim.compute_and_apply_stats(joint_fisher_loss, var_list=params)
            train_op, q_runner = optim.apply_gradients(list(zip(grads, params)))
        self.q_runner = q_runner
        self.lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = self.lr.value()

            td_map = {train_model.X: obs, A: actions, ADV: advs, R: rewards, PG_LR: cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks

            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, train_op],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.save = save
        self.load = load
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        tf.global_variables_initializer().run(session=sess)


class Runner(object):
    def __init__(self, env, model, nsteps, nstack, gamma, callback=None):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps, nh, nw, nc * nstack)
        self.obs = np.zeros((nenv, nh, nw, nc * nstack), dtype=np.uint8)
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
        self.total_reward = [0.0 for _ in range(nenv)]
        self.episode_rewards = [0.0]
        self.episodes = 0
        self.steps = 0
        self.callback = callback

    def update_obs(self, obs):
        self.obs = np.roll(self.obs, shift=-1, axis=3)
        self.obs[:, :, :, -1] = obs[:, :, :, 0]

    def run(self):
        model = self.model
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, states = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            self.states = states
            self.dones = dones
            self.steps += self.env.num_envs

            for n, done in enumerate(dones):
                self.total_reward[n] += float(rewards[n])
                if done:
                    self.episodes += 1
                    num_episodes = self.episodes
                    self.obs[n] = self.obs[n] * 0
                    self.episode_rewards.append(self.total_reward[n])
                    mean_100ep_reward = round(np.mean(self.episode_rewards[-101:-1]), 1)
                    # print("self.episode_rewards[-101:-1] %s" % self.episode_rewards[-101:-1])
                    print("env %s done! reward : %s mean_100ep_reward : %s " % (
                    n, self.total_reward[n], mean_100ep_reward))
                    logger.record_tabular("reward", self.total_reward[n])
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("steps", self.steps)
                    logger.record_tabular("episodes", self.episodes)
                    logger.dump_tabular()

                    if self.callback is not None:
                        self.callback(locals(), globals())

                    self.total_reward[n] = 0
            self.update_obs(obs)
            mb_rewards.append(rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values


def load(policy, env, seed, filename, total_timesteps=int(40e6), nprocs=32, nsteps=20,
         nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
         kfac_clip=0.001, lrschedule='linear'):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = 1
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs,
                               nsteps=nsteps, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef,
                               vf_fisher_coef=vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm,
                               kfac_clip=kfac_clip, lrschedule=lrschedule)
    model = make_model()

    model.load(filename)

    return model


def learn(policy,
          env,
          seed,
          total_timesteps=int(40e6),
          gamma=0.99,
          log_interval=1,
          nprocs=32,
          nsteps=20,
          nstack=4,
          ent_coef=0.01,
          vf_coef=0.5,
          vf_fisher_coef=1.0,
          lr=0.25,
          max_grad_norm=0.5,
          kfac_clip=0.001,
          save_interval=None,
          lrschedule='linear',
          callback=None):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    make_model = lambda: Model(policy,
                               ob_space,
                               ac_space,
                               nenvs,
                               total_timesteps,
                               nprocs=nprocs,
                               nsteps=nsteps,
                               nstack=nstack,
                               ent_coef=ent_coef,
                               vf_coef=vf_coef,
                               vf_fisher_coef=vf_fisher_coef,
                               lr=lr,
                               max_grad_norm=max_grad_norm,
                               kfac_clip=kfac_clip,
                               lrschedule=lrschedule)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()

    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma, callback=callback)
    nbatch = nenvs * nsteps
    tstart = time.time()
    enqueue_threads = model.q_runner.create_threads(model.sess, coord=tf.train.Coordinator(), start=True)
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
        model.old_obs = obs
        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("steps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("policy_loss", float(policy_loss))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()

        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            savepath = osp.join(logger.get_dir(), 'checkpoint%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)

    env.close()


def train_acktr(env_id, num_timesteps, seed, num_cpu, learning_rate):
    """Train a acktr model.

      Parameters
      -------
      env_id: environment to train on
      num_timesteps: int
          number of env steps to optimizer for
      seed: int
          number of random seed
      num_cpu: int
          number of parallel agents

      """
    num_timesteps //= 4

    def make_env(rank):
        def _thunk():
            # 1. Create gym environment
            env = gym.make(env_id)
            env.seed(seed + rank)
            if logger.get_dir():
                env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
            gym.logger.setLevel(logging.WARN)
            # 2. Apply action space wrapper
            env = MarioActionSpaceWrapper(env)
            # 3. Apply observation space wrapper to reduce input size
            env = ProcessFrame84(env)

            return env

        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    policy_fn = CnnPolicy
    learn(policy_fn, env, seed, total_timesteps=num_timesteps,
          nprocs=num_cpu, save_interval=True, lr=learning_rate,
          callback=acktr_callback)
    env.close()


def acktr_callback(locals, globals):
    global max_mean_reward, last_filename

    if ('mean_100ep_reward' in locals
        and locals['num_episodes'] >= 10
        and locals['mean_100ep_reward'] > max_mean_reward
        ):
        print("mean_100ep_reward : %s max_mean_reward : %s" %
              (locals['mean_100ep_reward'], max_mean_reward))

        if not os.path.exists(os.path.join(settings.ROOT_DIR, 'models/acktr/')):
            try:
                os.mkdir(os.path.join(settings.ROOT_DIR, 'models/'))
            except Exception as e:
                print(str(e))
            try:
                os.mkdir(os.path.join(settings.ROOT_DIR, 'models/acktr/'))
            except Exception as e:
                print(str(e))

        if last_filename != "":
            os.remove(last_filename)
            print("delete last model file : %s" % last_filename)

        max_mean_reward = locals['mean_100ep_reward']
        model = locals['model']

        filename = os.path.join(settings.ROOT_DIR, 'models/acktr/mario_reward_%s.pkl' % locals['mean_100ep_reward'])
        model.save(filename)
        print("save best mean_100ep_reward model to %s" % filename)
        last_filename = filename
