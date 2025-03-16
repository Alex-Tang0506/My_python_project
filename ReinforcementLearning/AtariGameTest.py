'''
gym for the environment, torch for the neural networks, optimization, automatic differentiation, utilities for vision tasks
'''

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from gym.utils.play import play
import pygame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import gym

env = gym.make("LunarLander-v2", render_mode="human")
# print(env.action_space)
# print(env.observation_space)
observation, info = env.reset(seed=42, return_info=True)

observation, info = env.reset(seed=42, return_info=True)
for i in range(1000):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation, info = env.reset(return_info=True)
env.close()
mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1, (pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make('LunarLander-v2', keys_to_action=mapping))


# Process for RL to play Atari game
'1. Choose a game, get the initial settings, in other word, s1(environment) and a1(action)'
'2. Q-learning, choose an action and calculate the reward, r1'
'3. Q-learning, simulate the next environment s2, base on the reward value r2 to update the Q-table'
'4. Iterate the process above, until the whole loop ended'


