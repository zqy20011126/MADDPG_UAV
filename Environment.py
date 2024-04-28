#!/usr/bin/env python
# coding: utf-8

# ### packages

# In[12]:


###########################################################################################

import numpy as np

import pandas as pd

###########################################################################################

import math

import random

import math

###########################################################################################

import matplotlib.pyplot as plt 

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

###########################################################################################

from shapely.geometry import Point

###########################################################################################

import gym

from gym import spaces

from gym.utils import seeding

###########################################################################################

import copy


# ### environment parameters

# In[13]:


###########################################################################################
###########################################################################################

scale=10

###########################################################################################
###########################################################################################

predators_num=3

gauards_num=3

learder_num=1

###########################################################################################
###########################################################################################
np.random.uniform(0,scale,2)


# ### map example

# In[27]:





# ### environment

# In[55]:


class Agent(object):
    
    def __init__(self,x,y,type_,velocity=3):
        
        self.x=x
        
        self.y=y
        
        self.type_=type_
        
        self.velocity=velocity

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

class Environment(gym.Env):

    def __init__(self,scale=10,
                 predators_num=3,\
                 gauards_num=2,\
                 learder_num=1):
        
        ###########################################################################################
        
        self.scale=scale

        self.predators_num=predators_num
        
        self.gauards_num=gauards_num
        
        self.learder_num=learder_num
        
        ###########################################################################################
        
    def reset(self):
        
        self.t=0
        
        ###########################################################################################
        
        self.agents=list()
        
        ###########################################################################################
        
        for i in range(self.predators_num):
            
            x,y=np.random.uniform(1,self.scale-1,1)[0],np.random.uniform(1,self.scale/2,1)[0]
            
            self.agents.append(Agent(x,y,'predator',velocity=1.5))
            
        ###########################################################################################
        
        for j in range(self.gauards_num):
            
            x,y=np.random.uniform(1,self.scale-1,1)[0],np.random.uniform(self.scale/2,self.scale-1,1)[0]
            
            self.agents.append(Agent(x,y,'gauard',velocity=2))
            
        ###########################################################################################
        
        for k in range(self.learder_num):
            
            x,y=np.random.uniform(1,self.scale-1,1)[0],np.random.uniform(self.scale/2,self.scale-1,1)[0]
            
            self.agents.append(Agent(x,y,'leader',velocity=1))
            
        ###########################################################################################
            
        obs=list()

        for i in range(len(self.agents)):

            obs.append(self.get_observation(i))
            
        return obs
            
    def get_observation(self,agent_id):
        
        ###########################################################################################
        
        current_agent=self.agents[agent_id]
        
        ###########################################################################################
        
        poses=list() # normalization
        
        for i in range(len(self.agents)):
            
            other_agent=self.agents[i]
            
            poses.append(other_agent.x/self.scale)
            
            poses.append(other_agent.y/self.scale)
            
        ###########################################################################################
        
        distances=list() # normalization
        
        for i in range(len(self.agents)):
            
            other_agent=self.agents[i]
            
            dis=Point(other_agent.x,other_agent.y).distance(Point(current_agent.x,current_agent.y))
            
            dis=dis/(self.scale*np.sqrt(2))
            
            distances.append(dis)
            
        ###########################################################################################
            
        return np.array(poses+distances)
    
    def step(self,actions):

        ###########################################################################################

        rewards=list()

        dones=list()

        ###########################################################################################

        self.t+=1

        ###########################################################################################

        for i in range(len(self.agents)):

            done=False

            ###########################################################################################

            action=actions[i]

            agent=self.agents[i]

            ###########################################################################################

            x_step=action[0]*np.cos(action[1])

            y_step=action[0]*np.sin(action[1])

            ###########################################################################################

            reward=0

            ###########################################################################################
            
            distances_to_gauard=[]
            
            if agent.type_=='predator':
                
                for j in range(len(self.agents)):
                    
                    other_agent=self.agents[j]
                    
                    if other_agent.type_=='gauard':
                    
                        distances_to_gauard.append(Point(agent.x,agent.y).distance(Point(other_agent.x,other_agent.y)))
                                                   
            ###########################################################################################
                                                   
            if len(distances_to_gauard)==0:
                
                self.agents[i].x=np.clip(agent.x+x_step,0,self.scale)

                self.agents[i].y=np.clip(agent.y+y_step,0,self.scale)
                
            elif min(distances_to_gauard)>0.1:
                
                self.agents[i].x=np.clip(agent.x+x_step,0,self.scale)

                self.agents[i].y=np.clip(agent.y+y_step,0,self.scale)

            ###########################################################################################

            distances_to_predator=[]

            distances_to_gauard=[]

            distances_to_leader=[]

            agent=self.agents[i]

            for j in range(len(self.agents)):

                other_agent=self.agents[j]

                if other_agent.type_=='predator':

                    distances_to_predator.append(Point(agent.x,agent.y).distance(Point(other_agent.x,other_agent.y)))

                elif other_agent.type_=='gauard':

                    distances_to_gauard.append(Point(agent.x,agent.y).distance(Point(other_agent.x,other_agent.y)))

                elif other_agent.type_=='leader':

                    distances_to_leader.append(Point(agent.x,agent.y).distance(Point(other_agent.x,other_agent.y)))

            ###########################################################################################

            if agent.type_=='predator':

                reward=-1*min(distances_to_leader)

            elif agent.type_=='gauard':

                reward=-1*min(distances_to_predator)

            elif agent.type_=='leader':

                reward=min(distances_to_predator)

            ###########################################################################################

            rewards.append(reward)

            ###########################################################################################

            if self.t>100:

                done=True

            dones.append(done)

        ###########################################################################################

        obs=list()

        for i in range(len(self.agents)):

            obs.append(self.get_observation(i))

        ###########################################################################################

        return obs, rewards, dones, {}
    
    def sample_action(self):
        
        actions=list()

        for i in range(len(self.agents)):
            
            velocity=np.random.uniform(0,self.agents[i].velocity)
            
            theta=np.random.uniform(0,2*np.pi)
            
            actions.append((velocity,theta))
            
        return actions
            
            
    
    def render(self):

        if self.t==1:

            fig, self.axs = plt.subplots(figsize = (10,10))

            labels = self.axs.get_xticklabels() + self.axs.get_yticklabels()

            [label.set_fontsize(20) for label in labels]

            [label.set_fontname('Arial') for label in labels]

            self.axs.grid()

            ###########################################################################################
            ###########################################################################################

            self.axs.set_ylim([0,self.scale])

            self.axs.set_xlim([0,self.scale])

            self.axs.set_xlabel('x',font_label)

            self.axs.set_ylabel('y',font_label)

            majorLocator   = MultipleLocator(2)

            self.axs.xaxis.set_major_locator(majorLocator)

            self.axs.yaxis.set_major_locator(majorLocator)

            ###########################################################################################
            ###########################################################################################
            
            self.circles={i:1 for i in range(len(self.agents))}

            for i in range(len(self.agents)):

                agent=self.agents[i]

                coords=[agent.x,agent.y]

                if agent.type_=='predator':

                    self.circles[i] = plt.Circle(coords, .5,color='red',alpha=.5)

                elif agent.type_=='gauard':

                    self.circles[i] = plt.Circle(coords, .5,color='blue',alpha=.5)

                elif agent.type_=='leader':

                    self.circles[i] = plt.Circle(coords, .5,color='green',alpha=.5)

                self.axs.add_artist(self.circles[i])

            plt.tight_layout()
            
        else:

            plt.pause(.1)
            
            for i in range(len(self.agents)):
                
                self.circles[i].remove()
                
            for i in range(len(self.agents)):
                
                agent=self.agents[i]

                coords=[agent.x,agent.y]

                if agent.type_=='predator':

                    self.circles[i] = plt.Circle(coords, .5,color='red',alpha=.5)

                elif agent.type_=='gauard':

                    self.circles[i] = plt.Circle(coords, .5,color='blue',alpha=.5)

                elif agent.type_=='leader':

                    self.circles[i] = plt.Circle(coords, .5,color='green',alpha=.5)

                self.axs.add_artist(self.circles[i])


