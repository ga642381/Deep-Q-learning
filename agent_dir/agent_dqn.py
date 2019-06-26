from agent_dir.agent import Agent
import os
import math
import random
import numpy as np
import pickle
import torch
import torch.nn as nn
import sys
from collections import namedtuple
from itertools import count
from torchsummary import summary
from collections import deque


# ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
ACTION_SIZE = 4

Transition = namedtuple('Transition',
                ('state', 'action', 'next_state', 'reward')) 

#In memory state size : (84, 84, 4)
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """ Saves a transition """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)
    
    
        

class DQN(nn.Module):    
    #dqn : input_size is (batch_size, 84, 84, 4) in CHW format
    def __init__(self, input_size=(84, 84, 4), action_size = ACTION_SIZE):
        super(DQN,self).__init__()
        
        (self.H, self.W, self.C)= input_size
        self.action_size = action_size
        
        self.CONVS = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2),
            nn.ReLU(),            
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2),
            nn.ReLU()
        )
        
        
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1        
        convH = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.H, 3, 2), 3, 2), 3, 2), 3, 2)
        convW = conv2d_size_out(conv2d_size_out(conv2d_size_out(conv2d_size_out(self.W, 3, 2), 3, 2), 3, 2), 3, 2)
        self.linear_input_size = convW * convH * 128
        
        
        self.FCS = nn.Sequential(
            nn.Linear(self.linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)             
        )
        
        
    def forward(self, observation): 
        observation = observation.permute(0, 3, 1, 2)
        observation = self.CONVS(observation)                
        observation = observation.view(-1, self.linear_input_size)
        actionsQ = self.FCS(observation)        
        return actionsQ 
        

"""
    in main:
        env = Environment(env_name, args, atari_wrapper=True)
        agent = Agent_DQN(env, args)
    
    
    for deep q learning:
        observation size is (84, 84, 4) : 4 gray-scale frames stacked
    
    for the atari game breakout:
        use self.env.env.unwrapped.get_action_meanings() to get the action space
        action space : ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
        action size : 4
"""


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
            class Agent(object):
                def __init__(self, env):
                    self.env = env
                    
            self.env means Environment, containing 6 functions:
                1. seed(seed)
                2. reset()
                3. step(action)
                4. get_action_space()
                5. get_observation_space()
                6. get_random_action()        
                
        """
        
        super(Agent_DQN,self).__init__(env)
        self.DQN_INPUT_SIZE = (84, 84, 4)
        self.BATCH_SIZE = 64
        self.GAMMA = 0.99
        self.EPS_START = 1
        self.EPS_DECAY = 200000
        self.EPS_END = 0.02
        
        self.episodes_done = 0
        self.steps_done = 0
        
        
        self.device = torch.device('cuda')        
        self.Q_policy = DQN(self.DQN_INPUT_SIZE).to(self.device)     
        self.Q_target = DQN(self.DQN_INPUT_SIZE).to(self.device)
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        self.Q_target.eval()
        self.memory = ReplayMemory(20000)
        
        self.RewardQueue = deque(maxlen=50)
        self.AverageReward_hist = []
        
        self.optimizer = torch.optim.Adam(self.Q_policy.parameters(), lr=1.5e-4)    
        self.MSE_loss = nn.MSELoss().to(self.device)
        """------------------------------------------------------------------"""
        
        if args.test_dqn:
            #you can load your model here
            self.load("Q_saved_base_v2", 24000)
            print('loading trained model')   
            
            
        print("Using Device : {}".format(self.device))
        print('---------- Networks architecture -------------')
        summary(self.Q_policy, (self.DQN_INPUT_SIZE))
        print('----------------------------------------------')
        
    def load(self, save_dir, i_episode):
        print("dir : {}  i_episode: {}".format(save_dir, i_episode))
        model_path = os.path.join(save_dir, str(i_episode) + "_Q.pkl")
        self.Q_policy.load_state_dict(torch.load(model_path))
        self.Q_target.load_state_dict(self.Q_policy.state_dict())
        
        with open (os.path.join(save_dir, str(i_episode) + ".pkl"), 'rb') as f:
            Data = pickle.load(f)
            self.steps_done = int(Data[0])
            self.episodes_done = int(Data[1])

    def init_game_setting(self):
        print("Q_policy_eval()")
        self.Q_policy.eval()
        pass
    
    def train(self):
        print("Training start!!!! ")
        
        # observation shape : (84, 84, 4)
        # -> transpose((2, 0, 1))
        # -> shape : (4, 84, 84)
        
        def save(save_dir, i_episode, dump_data):
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.Q_policy.state_dict(), os.path.join(save_dir, str(i_episode) + "_Q.pkl"))
            with open(os.path.join(save_dir, '{}.pkl'.format(i_episode)), 'wb') as f:
                pickle.dump(dump_data, f)
        
        #######################################################################
        #                       MAIN TRAINING LOOP                            # 
        #######################################################################
        
        #self.load("Q_saved_double_v3", 13000)                   
        for e in count():
            state = self.env.reset()
            REWARD = 0
            
            for s in count():
                # in make_action self.step_done += 1
                # IMPORTANT! : make_action receive size (84,84,4)
                action = self.make_action(state, False)
                next_state, reward, done, _ = self.env.step(action)
                
                if not done:
                    self.memory.push(state, [action], next_state, [reward])
                    state = next_state                      
                
                if self.steps_done % 4 ==0:
                    self.optimize_model()
                
                if self.steps_done % 1000 == 0:
                    print("Q_target <- Q_policy")
                    self.Q_target.load_state_dict(self.Q_policy.state_dict())         
                
                    
                REWARD = REWARD + reward
                if done:
                    self.RewardQueue.append(REWARD)                    
                    average_reward = sum(self.RewardQueue) / len(self.RewardQueue)
                    self.AverageReward_hist.append(average_reward)
                    print("episode : {}, step : {}, average_reward:{}".format(self.episodes_done, self.steps_done, average_reward))                    
                    break        
                
            if self.episodes_done % 1000 == 0:
                dump_data = [self.steps_done, self.episodes_done, self.AverageReward_hist]
                print("episode : ", self.episodes_done, "saving model...")
                save("Q_saved_base_final", self.episodes_done, dump_data)
                
                
            self.episodes_done += 1
            
            
                
    def make_action(self, observation, test=True):
        """
            Return predicted action of your agent
    
            Input:
                observation: np.array
                    stack 4 last preprocessed frames, shape: (84, 84, 4)
    
            Return:
                action: int
                    the predicted action from trained model
        """
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)                            
        self.steps_done += 1
        
        if test:
            if np.random.rand() < 0.025 :
                return self.env.get_random_action()

            else:
                with torch.no_grad():
                    observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    actionsQ = self.Q_policy(observation)
                    return torch.argmax(actionsQ).item()
        
        # epsilon greedy
        else : 
            if np.random.rand() < eps_threshold:
                """ random """
                return self.env.get_random_action()
            
            
            else:
                """ greedy """
                # input is (84, 84, 4)
                # permute -> (4, 84, 84)
                # unsqueeze -> (1, 4, 84, 84)
                
                with torch.no_grad():
                    observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
                    actionsQ = self.Q_policy(observation)
                    return torch.argmax(actionsQ).item()
                
            
    def optimize_model(self):
        # there should be enough data in memory        
        if len(self.memory) < self.BATCH_SIZE:
            return 
        
        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        # zip(*[('a', 1), ('b', 2), ('c', 3), ('d', 4)])
        # ->[('a', 'b', 'c', 'd'), (1, 2, 3, 4)]
        # converts batch-array of Transitions to Transition of batch-arrays 

        def to_tuple_of_tensor(t):            
            return(tuple(torch.Tensor(e).unsqueeze(0) for e in list(t)))
        
        # 1. batch.next_state 
        next_state_batch = torch.cat(to_tuple_of_tensor(batch.next_state)).float().to(self.device)
        
        # 2. batch.state
        state_batch = torch.cat(to_tuple_of_tensor(batch.state)).float().to(self.device)

        # to long is for gather
        # 3. batch.action (32, 1)
        action_batch = torch.cat(to_tuple_of_tensor(batch.action)).to(self.device).long()

        # 4. batch.reward (32, 1)
        reward_batch = torch.cat(to_tuple_of_tensor(batch.reward)).to(self.device)
        
        Qvalue_t0 = self.Q_policy(state_batch).gather(1, index=action_batch)
        Qvalue_t1 = self.Q_target(next_state_batch).max(1)[0].unsqueeze(1).detach()
        expected_Qvalue = (Qvalue_t1*self.GAMMA) + reward_batch
        
        
        """
        Qvalue_t1_a = self.Q_policy(next_state_batch).max(1)[1].unsqueeze(1).long()        
        double_Qvalue_t1 = self.Q_target(next_state_batch).gather(1, index=Qvalue_t1_a)
        expected_Qvalue = (double_Qvalue_t1*self.GAMMA) + reward_batch
        """
        self.optimizer.zero_grad()
        loss = self.MSE_loss(Qvalue_t0, expected_Qvalue)
        loss.backward()
        for param in self.Q_policy.parameters():
            param.grad.data.clamp_(-1, 1)             
        
        self.optimizer.step()
    
       
    
    
