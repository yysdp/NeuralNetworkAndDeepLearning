

import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from DelayDDPG import DDPG
from gym import spaces 

#np.set_printoptions(precision=5)

env = gym.make('BipedalWalkerHardcore-v3')
env = env.unwrapped

env.seed(0)
np.random.seed(0)
ddpg = DDPG(state_size = env.observation_space.shape[0],action_size = env.action_space.shape[0],
            action_low = env.action_space.low,action_high = env.action_space.high,
            memory_size = 1000000,batch_size = 128,epsilon = 1,
            discount = 0.99,gama = 0.005,act_noise=0.3,lr=1e-4)

#ddpg.laod_memory()
##接力训练的要求
#ddpg.laod_m_count()
ddpg.load_model_weights()
save_freq=1
steps_per_epoch=2000
epochs=200
epoch_step = 0
total_step = 0
ep_rw = 0
#用于画图
y_rw = []
x_epochs =  np.linspace(1, epochs,epochs)
observation = env.reset()
for e in range(epochs):
    epoch_step = 0
    ep_rw = 0
    observation = env.reset()
    for t in range(steps_per_epoch):
        env.render()
        action = ddpg.get_action(observation)
        _observation, reward, done, info = env.step(action)
        #格式化
        if done:
            d = 1
            reward = 0.0
        else:
            d = 0
        #保存记录
        ddpg.store_transition(observation,action,reward*5,d,_observation)
        #reward = float('%.5f' % reward)
        ep_rw = ep_rw + reward        
        total_step += 1
        epoch_step += 1
        if done or t == (steps_per_epoch - 1):
            print('epoceh',e,'reward',ep_rw,'steps',epoch_step)
            y_rw.append(ep_rw)
            break
        observation = _observation
    #    if(total_step >=0):
    #        batch_memory =ddpg.sample_minibatch_transitions()
    #        #得到 y(q_target_value)
    #        s,a,y = ddpg.set_q_target_value(batch_memory)
    #        ddpg.update_critic(s,a,y)
    #        ddpg.update_actor(s)
    #        ddpg.update_target_network()
    #if (e+1) % save_freq == 0:
    #    ddpg.save_model_weights()
    #if((e+1) % (save_freq * 50)) == 0:
    #        ddpg.save_memery()
    #        ddpg.save_m_count()
    #        print('m_count',ddpg.m_count)
#画图，x为第几回合，y为每回合得到的奖励
print('m_count',ddpg.m_count)
plt.plot(x_epochs,y_rw,label='evey_epoch_reward')
plt.show()
env.close()
        


