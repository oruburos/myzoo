import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.utils import average_total_reward
import gerrymandering
import Gerry2
from pettingzoo import *
import imageio


def start(name):
    print("env")
    env = gerrymandering.env()
    env2 = Gerrry2.env()

    env.reset()
    obs_list = []
    iteration = 0
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        # print(observation)
        # print(reward)
        action = random.randrange(0,2)
        iteration = iteration + 1
        if not done:
            env.step(action)
        else:
            env.step(None)

        frame = env.render()
      #  obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)));

    #plt.imshow(frame)
    #plt.axis('off')
    #plt.show()

    print("frames", len(obs_list))
    print("observations", iteration)

    #imageio.mimsave('omar.gif', obs_list)

    env.close()

if __name__ == '__main__':
    start('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
