import gym

from wrappers import MarioActionSpaceWrapper, ProcessFrame84


class RandomAgent(object):
    @staticmethod
    def act(action_space, observation, reward, done):
        return action_space.sample()


def run_random_agent(env_id, episodes):
    episode_count = episodes
    reward = 0
    done = False
    agent = RandomAgent()

    for i in range(episode_count):
        env = gym.make(env_id)
        env = MarioActionSpaceWrapper(env)
        env = ProcessFrame84(env)

        ob = env.reset()
        while True:
            action = agent.act(env.action_space, ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
