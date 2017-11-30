import gym

from wrappers import ToDiscreteWrapper, ProcessFrame84


class RandomAgent(object):
    @staticmethod
    def act(action_space):
        return action_space.sample()


def run_random_agent(env_id, episodes):
    done = False
    agent = RandomAgent()

    for i in range(episodes):
        env = gym.make(env_id)
        env = ToDiscreteWrapper(env)
        env = ProcessFrame84(env) # [1, 84, 84, 3]

        env.reset()
        while True:
            action = agent.act(env.action_space)
            _, reward, done, _ = env.step(action)
            print('reward: {}'.format(reward))
            if done:
                break