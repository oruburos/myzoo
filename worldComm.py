import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pettingzoo.utils import average_total_reward
#from pettingzoo.mpe import simple_world_comm_v2
import Gerry2
from pettingzoo import *
import imageio

print("env")
env = Gerry2.env()
env.reset()
obs_list = []
iteration = 0
for agent in env.agent_iter():
    observation, reward, done, info = env.last()

    action = iteration % 4
    iteration = iteration + 1
    if not done:
        env.step(action)

    else:
    #    print(observation)
    #    print(reward)
    #    print(info)
        env.step(None)

    frame = env.render(mode='rgb_array')
    #obs_list.append(np.transpose(env.render(mode='rgb_array'), axes=(1, 0, 2)));
    obs_list.append( env.render(mode='rgb_array'));
#plt.imshow(frame)
#plt.axis('off')
#plt.show()

print("frames", len(obs_list))
print("observations", iteration)

imageio.mimsave('omar2.gif', obs_list)

env.close()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
