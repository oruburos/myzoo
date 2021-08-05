import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.utils import average_total_reward
from pettingzoo.mpe import simple_world_comm_v2
from pettingzoo import *
import imageio


def start(name):
    print("env")



    env = simple_world_comm_v2.env(num_good=2, num_adversaries=2, num_obstacles=1,
                                   num_food=2, max_cycles=25, num_forests=2, continuous_actions=False)

    env.reset()
    obs_list = []
    iteration = 0
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()
        # print(observation)
        # print(reward)
        action = 1
        iteration = iteration + 1
        if not done:
            env.step(action)
        else:
            env.step(None)

        frame = env.render(mode='rgb_array')
        obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)));

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
