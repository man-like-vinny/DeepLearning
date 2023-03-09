import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.activations import softmax
from keras.optimizers import Adam
import os
import time
import numpy as np

# def softmax(x, temperature=0.025): 
#     """Compute softmax values for each sets of scores in x."""
#     x = (x - np.expand_dims(np.max(x, 1), 1))
#     x = x/temperature    
#     e_x = np.exp(x)
#     return e_x / (np.expand_dims(e_x.sum(1), -1) + 1e-5)

def softmax(x, temperature=0.025): 
    """Compute softmax values for each sets of scores in x."""
    if len(x.shape) == 1:
      x = x - np.max(x)
      e_x = np.exp(x / temperature)
      return e_x / np.sum(e_x, axis=0)
    else:
      x = (x - np.expand_dims(np.max(x, 1), 1))
      x = x/temperature    
      e_x = np.exp(x)
      return e_x / (np.expand_dims(e_x.sum(1), -1) + 1e-5)

class DQNAgent:
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.memory = deque(maxlen=20000)
    self.gamma = 0.95    # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.model = self._build_model()

    
  def _build_model(self):
    # Neural Net for Deep-Q learning Model
    model = Sequential()
    model.add(Dense(24, input_dim=self.state_size, activation='relu'))
    model.add(Dense(48, activation='relu'))
    model.add(Dense(self.action_size, activation='linear'))
    model.compile(loss='mse',
                  optimizer=Adam(lr=self.learning_rate))
    return model

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def act(self, state):# We implement the epsilon-greedy policy
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0]) # returns action

    #softmax policy
    # act_values = self.model.predict(state)[0]
    # probabilities = softmax(act_values,temperature=0.025)
    # action = np.random.choice(self.action_size, p=probabilities)
    # return action 

  def exploit(self, state): # When we test the agent we dont want it to explore anymore, but to exploit what it has learnt
    act_values = self.model.predict(state)
    return np.argmax(act_values[0]) 

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    
    state_b = np.squeeze(np.array(list(map(lambda x: x[0], minibatch))))
    action_b = np.squeeze(np.array(list(map(lambda x: x[1], minibatch))))
    reward_b = np.squeeze(np.array(list(map(lambda x: x[2], minibatch))))
    next_state_b = np.squeeze(np.array(list(map(lambda x: x[3], minibatch))))
    done_b = np.squeeze(np.array(list(map(lambda x: x[4], minibatch))))

    ### Q-learning
    # target = (reward_b + self.gamma * np.amax(self.model.predict(next_state_b), 1))
    # target[done_b==1] = reward_b[done_b==1]
    # target_f = self.model.predict(state_b)

    # for k in range(target_f.shape[0]):
    #   target_f[k][action_b[k]] = target[k]
    # self.model.train_on_batch(state_b, target_f)
    # if self.epsilon > self.epsilon_min:
    #     self.epsilon *= self.epsilon_decay

    ### SARSA
    #In this modified function, for each batch sample, we compute the next action based on the next state by calling the act method with the next state as input. Then we compute the target using the next state and the next action, and update the target for the current state and action accordingly. Finally, we use the updated targets to train the model.
    #Note that in SARSA, we choose the next action based on the current policy, rather than choosing the maximum action value like in Q-learning.
    target = np.zeros((batch_size, ))
    for i in range(batch_size):
      if not done_b[i]:
        next_action = self.act(np.expand_dims(next_state_b[i], axis=0))
        target[i] = reward_b[i] + self.gamma * self.model.predict(np.expand_dims(next_state_b[i], axis=0))[0][next_action]
      else:
        target[i] = reward_b[i]

    target_f = self.model.predict(state_b)

    for k in range(target_f.shape[0]):
      target_f[k][action_b[k]] = target[k]

    self.model.train_on_batch(state_b, target_f)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)
  def save(self, name):
    self.model.save_weights(name)


EPISODES = 200
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
batch_size = 32
episode_reward_list = deque(maxlen=50)

for e in range(EPISODES):
  state = env.reset()
  state = np.reshape(state, [1, state_size])
  total_reward = 0
  for time in range(200):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    next_state = np.reshape(next_state, [1, state_size])
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    if done:
      break
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)
  episode_reward_list.append(total_reward)
  episode_reward_avg = np.array(episode_reward_list).mean()
  print("episode: {}/{}, score: {}, e: {:.2}, last 50 ep. avg. rew.: {:.2f}".format(e, EPISODES, total_reward, agent.epsilon, episode_reward_avg))      