from agent_dir.agent import Agent
# import scipy
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchsummary import summary
from tqdm import tqdm

def prepro(o,image_size=(80,80)):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = o.astype(np.uint8)
    #resized = scipy.misc.imresize(y, image_size)
    #return np.expand_dims(resized.astype(np.float32),axis=2)
    resized = cv2.resize(y, image_size, interpolation=cv2.INTER_CUBIC)
    resized = resized.astype(np.float32)/255.
    rgb = np.swapaxes(np.swapaxes(resized, 1, 2), 0,1)
    grey = 0.299*rgb[0]+0.587*rgb[1]+0.114*rgb[2]
    return grey


class AgentNN(torch.nn.Module):
    def __init__(self, input_size, input_dim=1, output_class=3):
        super(AgentNN,self).__init__()
        self.input_dim = input_dim
        self.output_class = output_class
        self.input_size = input_size
        
        self.fully = nn.Sequential(
            # [batch, 16,20,20]
            nn.Linear(input_size[0]*input_size[1]*input_size[2], 200),
            nn.ReLU(),
            nn.Linear(200, self.output_class),
            nn.Softmax(-1)
        )

        
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        out = self.fully(x.view(batch_size, -1))
        return out
        


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)

        if args.test_pg:
            #you can load your model here
            print('loading trained model')

        ##################
        # YOUR CODE HERE #
        ##################
        self.lastframe = 0
        self.gamma = 0.99
        self.device = torch.device('cuda')
        self.agentnn = AgentNN((1,80,80), input_dim=1, output_class=3).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.agentnn.parameters(), lr=5e-3)
        #self.optimizer = torch.optim.Adam(self.agentnn.parameters(), lr=0.1)
        
        summary(self.agentnn, (1,80,80))


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        epochs = 1000
        round_episodes = 10
        save_rate = 20
        self.env.seed(6873)
        self.agentnn.train()
        
        for e in range(epochs):
            
            episode_losses = []
            avg_reward_prt = 0.0
                        
#             for i in tqdm(range(round_episodes)):
            for i in range(round_episodes):
                #playing one game
                
                current_rewards = []
                current_logprob = []
                
                state = self.env.reset()
                self.init_game_setting()
                done = False                
                
                while(not done):
                    logp_action, action = self.make_action(state, test=False)
                    state, reward, done, info = self.env.step(action)
                    current_rewards.append(reward)
                    current_logprob.append(logp_action)
                
                ### make rewards for each action
                current_rewards_adjust = []
                littleR = 0
                for r in current_rewards[::-1]:
                    littleR = r + self.gamma*littleR
                    current_rewards_adjust.append(littleR)
                current_rewards_adjust = current_rewards_adjust[::-1]
                avg_reward_prt += sum(current_rewards)
                
#                 mean = sum(current_rewards_adjust) / (len(current_rewards_adjust)+1e-8)
                rewardTensor =  torch.FloatTensor(current_rewards_adjust).to(self.device)
                r_mean = rewardTensor.mean(-1)
                r_std = rewardTensor.std(-1)
                rewardTensor = (rewardTensor - r_mean)/r_std
#                 losses = 0
#                 for lp,r in zip(current_logprob, current_rewards_adjust):
#                     losses += lp*(r-mean)
                losses = torch.sum(torch.mul(torch.stack(current_logprob,0), rewardTensor), -1)                
                episode_losses.append(losses)
                
            self.optimizer.zero_grad()
            final_loss = - torch.stack(episode_losses, 0).mean(-1)    
#             print(final_loss.requires_grad)
            final_loss.backward()
            self.optimizer.step()
            
            print("epoch {}: average reward: {} loss: {}".format(e+1, avg_reward_prt/round_episodes, final_loss.item()), end='\n')
            if e % save_rate == 0:
                torch.save(self.agentnn, "checkpoint/Agent"+str(e))
#             for i,param in enumerate(self.agentnn.parameters()):
#                 if i == 2:
#                     print(param.data)

            
            
        


    def trim(self, screen):
        # (176,160,3)
        return screen[34:,:,:]
        
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        processed = prepro(self.trim(observation))
        
        
        residual = processed - self.lastframe
        self.lastframe = processed
                
        #import matplotlib.pyplot as plt
        #plt.subplot(1,3,1)
        #plt.imshow(np.swapaxes(np.swapaxes(self.lastframe, 0,1), 1,2))
        #plt.subplot(1,3,2)
        #plt.imshow(np.swapaxes(np.swapaxes(processed, 0,1), 1,2))
        #plt.subplot(1,3,3)
        #plt.imshow(np.swapaxes(np.swapaxes(residual, 0,1), 1,2))
        #plt.show()
        
        
        
        input_t = torch.tensor([residual]).to(self.device)
        input_t = input_t.view(1, 1, 80, 80)
        probs = self.agentnn.forward(input_t)[0]
        distri = torch.distributions.Categorical(probs)
        index = distri.sample()
        
        print(probs)
        
        
        #print(self.env.env.unwrapped.get_action_meanings())
        #['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
#         index = index.item()
#         print(index)

        actions = [0, 2, 3]
        
        if test:
            return actions[index.item()]
        else:
#             return torch.log(probs[index]+1e-8), index.item()
            return distri.log_prob(index), actions[index.item()]
