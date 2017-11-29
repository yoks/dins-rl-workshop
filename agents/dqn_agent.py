import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
from wrappers import MarioActionSpaceWrapper, ProcessFrame84
import tensorflow as tf


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=0.01, cpu=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5048)
        self.gamma = 0.9  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.device = '/cpu:0' if cpu else '/gpu:0'
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    @staticmethod
    def _huber_loss(target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return K.mean(K.sqrt(1 + K.square(error)) - 1, axis=-1)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model

        with tf.device(self.device):
            model = Sequential()
            model.add(Dense(128, input_dim=self.state_size, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(128, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss=self._huber_loss,
                          optimizer=Adam(lr=self.learning_rate))
            return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        minibatch = self.memory if len(self.memory) < 1024 else random.sample(self.memory, 1024)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train_dqn_agent(env_id, episodes):
    episode_count = episodes
    env = gym.make(env_id)
    env = MarioActionSpaceWrapper(env)
    env = ProcessFrame84(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    #agent.load("./model/dqn-mario_615.h5")
    for episode in range(episode_count):
        env = gym.make(env_id)
        env = MarioActionSpaceWrapper(env)
        env = ProcessFrame84(env)
        state = env.reset()
        state = np.reshape(state, [state_size, state_size])
        reward_sum = 0
        actions = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [state_size, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            reward_sum += reward
            actions += 1
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}, actions: {}"
                      .format(episode, episodes, reward_sum, agent.epsilon, actions))
                break
        agent.replay()
        if episode % 5 == 0:
            agent.save("./model/dqn-mario_{}.h5".format(episode))


def run_dqn_agent(env_id):
    env = gym.make(env_id)
    env = MarioActionSpaceWrapper(env)
    env = ProcessFrame84(env)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, epsilon=0.001, cpu=True)
    agent.load("./model/dqn-mario_835.h5")
    state = env.reset()
    state = np.reshape(state, [state_size, state_size])
    reward_sum = 0
    actions = 0
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [state_size, state_size])
        state = next_state
        reward_sum += reward
        actions += 1
        if done:
            agent.update_target_model()
            print("run finished: score: {}, e: {:.2}, actions: {}"
                  .format(reward_sum, agent.epsilon, actions))
            break


def visualize_layers():
    env = gym.make('ppaquette/SuperMarioBros-1-1-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, epsilon=0.001, cpu=True)
    agent.load("./model/dqn-mario_150.h5")
