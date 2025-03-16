import gym
import time
env = gym.make('CartPole-v1')
print(env.action_space)
print(env.observation_space)
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for i in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(i))
            break
env.close()
