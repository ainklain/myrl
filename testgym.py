import gym

env = gym.make('CarRacing-v0')

env.reset()
env.render()
for i in range(1000):
    env.step(env.action_space.sample())
    env.render()

env.close()