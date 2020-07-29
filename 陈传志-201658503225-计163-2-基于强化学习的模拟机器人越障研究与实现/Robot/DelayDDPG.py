
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float64') #设置神经网络层的数据类型
class DDPG:
    def __init__(self,state_size=0,action_size=0,action_low=0,
                 action_high=0,memory_size=0,batch_size=0,
                 epsilon=0,discount=0,gama=0,act_noise=0.1,lr=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.a_low = action_low
        self.a_high = action_high
        self.memory=np.ones((memory_size,state_size*2+action_size+1+1))
        self.batch_size=batch_size
        self.epsilon = epsilon #epsilon of action selection
        self.epsilon_min = 0.01 # min epsilon of ε-greedy.
        self.epsilon_decay = 0.995 #discount rate for epsilon.
        self.discount = discount
        self.gama = gama # 用于更新tagert network
        self.act_noise = act_noise
        self.lr = lr
        #建立神经网络
        self.actor_network = self.build_actor_network()
        self.critic_network = self.build_critic_network()
        self.actor_target_net = self.build_actor_network()
        self.critic_target_net = self.build_critic_network()
        self.actor_target_net.set_weights(self.actor_network.get_weights())
        self.critic_target_net.set_weights(self.critic_network.get_weights())
        self.actor_opt = keras.optimizers.Adam(learning_rate=lr)
        self.critic_opt = keras.optimizers.Adam(learning_rate=lr)
        self.memory_size = memory_size
        self.m_count = 0

    def store_transition(self,s,a,r,d,_s):
        transition = np.hstack((s,a,r,d,_s))
        index = self.m_count % self.memory_size
        self.memory[index,:] = transition
        self.m_count=self.m_count+1

    def save_memery(self):
        if self.m_count<self.memory_size:
            memory = self.memory[0:self.m_count,:]
        else:
            memory = self.memory
        np.savetxt('memory.txt',memory)
        print('save_memory')
        
    def laod_memory(self):
        if os.path.exists('memory.txt'):
            memory_txt = np.loadtxt('memory.txt')
            for i in range(memory_txt.shape[0]):
                self.memory[i] = memory_txt[i]
                self.m_count = self.m_count + 1
            print('m_count',self.m_count)
            print('laod_memory')

    def build_actor_network(self):
        input = keras.Input(shape=(self.state_size,))
        l1 = layers.Dense(256,activation='relu')(input)
        l2 = layers.Dense(512,activation='relu')(l1)
        output = layers.Dense(self.action_size,activation = 'tanh')(l2)
        model = keras.Model(input,output)
        return model

    def build_critic_network(self):
        state_input = keras.Input(shape=(self.state_size,))
        action_input = keras.Input(shape=(self.action_size,))
        state_layer = layers.Dense(40,activation='relu')(state_input)
        action_layer = layers.Dense(40,activation='relu')(action_input)
        x = layers.concatenate([state_layer,action_layer])
        layer2 = layers.Dense(256,activation='relu')(x)
        output = layers.Dense(1,activation = 'linear')(layer2)
        model = keras.Model([state_input,action_input],output)
        return model

    def Ornstein_Uhlenbeck(self,action,theta=0.2,sigma=0.25):
        u = np.zeros(self.action_size)
        return theta*(u-action)+np.random.randn(self.action_size)*sigma

    def update_epsilon(self):
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sample_minibatch_transitions(self):
        if self.m_count<self.memory_size:
            memory = self.memory[0:self.m_count,:]
        else:
            memory = self.memory
        #随机抽样
        row_rand_array = np.arange(memory.shape[0])
        np.random.shuffle(row_rand_array)
        batch_memory = memory[row_rand_array[0:self.batch_size]]
        return batch_memory

    def set_q_target_value(self,batch_memory):
        s_size = self.state_size
        a_size = self.action_size
        state = batch_memory[:,:s_size]
        action = batch_memory[:,s_size:s_size+a_size]
        reward = batch_memory[:,s_size+a_size]
        d = batch_memory[:,s_size+a_size+1]
        _state = batch_memory[:,-s_size:]
        _action = self.actor_target_net.predict(_state)
        _q_target_value = self.critic_target_net.predict([_state,_action])
        y= np.zeros(reward.size).reshape(reward.size,1)
        for i in range(reward.size):
            target_value = reward[i] + (1-d[i]) * self.discount * _q_target_value[i][0]
            y[i][0] = target_value
        return state,action,y

    def update_critic(self,state,action,y):

        with tf.GradientTape() as tape:
            q = self.critic_network([state, action])
            td_error = keras.losses.mean_squared_error(y, q)
        c_grads = tape.gradient(td_error, self.critic_network.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic_network.trainable_weights))
        
    def update_actor(self,state):
        
        with tf.GradientTape() as tap:
            action = self.actor_network(state)
            q = self.critic_network([state, action])
            a_loss = -tf.reduce_mean(q)
            #Q_function =  self.critic_network([state,self.actor_network(state)]) / -self.batch_size
        # 计算 dq/da * da/dtheta
        a_grad = tap.gradient(a_loss,self.actor_network.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grad, self.actor_network.trainable_weights))

    def update_target_network(self):
        actor_weights = self.actor_network.get_weights()
        actor_target_weights = self.actor_target_net.get_weights()
        critic_weights = self.critic_network.get_weights()
        critic_target_weights = self.critic_target_net.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = (1 - self.gama) * actor_target_weights[i] + self.gama * actor_weights[i]
        
        for i in range(len(critic_weights)):
            critic_target_weights[i] = (1 - self.gama) * critic_target_weights[i] + self.gama * critic_weights[i]
        
        self.actor_target_net.set_weights(actor_target_weights)
        self.critic_target_net.set_weights(critic_target_weights)
        
    def get_action(self,state):

        state = np.array(state).reshape(1,self.state_size)
        action = self.actor_network.predict(state)[0]
        #ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_size),x0 = action)
        ##print('epsilon',self.epsilon)
        #action = np.clip(np.random.normal(action, self.act_noise),self.a_low,self.a_high)
        return action
    
    def load_model_weights(self):

        if os.path.exists('./weights/actor_model.h5') and os.path.exists('./weights/critic_model.h5'):
            self.actor_network.load_weights('./weights/actor_model.h5')           
            self.critic_network.load_weights('./weights/critic_model.h5')
            print('load_weights')
        if  os.path.exists('./weights/actor_target_model.h5') and os.path.exists('./weights/critic_target_model.h5'):
            self.actor_target_net.load_weights('./weights/actor_target_model.h5')
            self.critic_target_net.load_weights('./weights/critic_target_model.h5')
            print('load_target.weights')
    def save_model_weights(self):
        self.actor_network.save_weights('./weights/actor_model.h5')
        self.actor_target_net.save_weights('./weights/actor_target_model.h5')
        self.critic_network.save_weights('./weights/critic_model.h5')
        self.critic_target_net.save_weights('./weights/critic_target_model.h5')
        print('save_weights')

    def save_m_count(self):
        # w, 只写模式 （文件不可读，如果文件不存在，则创建一个新的文件，如果文件存在，则会清空里面的内容）
        f = open('m_count.txt',mode='w',encoding='utf-8')
        f.write(str(self.m_count))
        f.close()

    def laod_m_count(self):
        #r, 只读的方式打开（文件必须存在，如果文件不存在，则会抛出异常）
        f = open('m_count.txt',mode='r',encoding='utf-8')
        self.m_count =int(f.read())
        print(self.m_count)
        f.close()

#reference :https://blog.csdn.net/qq_33254870/java/article/details/105137275
#奥恩斯坦-乌伦贝克
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu = 0, x0 = None, sigma=1.0, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

