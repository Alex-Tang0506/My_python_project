import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

'''Path setting'''
log_path = os.path.join('Training', 'Logs')
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Carpole')
print(log_path)
print(PPO_path)

'''Environment importing'''
env = gym.make('CartPole-v0')
env = DummyVecEnv([lambda: env])

'''Model training'''
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=100)
# model.save(PPO_path)
# evaluate_policy(model, env, n_eval_episodes=50, render=True)
# env.close()

'''Model testing'''
model = PPO.load(PPO_path, env=env)
obs = env.reset()
print(model.predict(obs))
episodes = 5
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs)  # Using the trained model here
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{}, Score:{}'.format(episode, score))
env.close()

'''Viewing in Tensorboard'''
training_log_path = os.path.join(log_path, 'PPO_11')
print(training_log_path)
