import gym

from wrappers import MarioActionSpaceWrapper, ProcessFrame84


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def run_random_agent(env, episodes):
    env = MarioActionSpaceWrapper(env)
    env = ProcessFrame84(env)

    agent = RandomAgent(env.action_space)

    episode_count = episodes
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
