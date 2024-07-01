import gymnasium
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

env = gymnasium.make("CartPole-v1")

class CNNQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(CNNQNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
model = CNNQNetwork(env.observation_space.shape[0], env.action_space.n)
target_model = CNNQNetwork(env.observation_space.shape[0], env.action_space.n)
          
optimizer = optim.Adam(model.parameters(), lr=0.0001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
target_model.to(device)

train_num = 1000
epsilode_length = 1000
target_network_replace_freq = 100
epsilon_values = [max(0.01, 0.05 - 0.005 * i) for i in range(train_num)]
memory = deque(maxlen=6000)
batch_size = 64
loss_func = nn.MSELoss()

def optimize_model(learn_step_counter):
    if len(memory) < batch_size:
        return learn_step_counter
    if learn_step_counter % target_network_replace_freq == 0:
        # Assign the parameters of eval_net to target_net
        param = model.state_dict()
        target_model.load_state_dict(param)
        
    learn_step_counter += 1
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones, truncated = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.bool).to(device)
    truncated = torch.tensor(truncated, dtype=torch.bool).to(device)
    
    actions = torch.reshape(actions, (-1, 1))
    rewards = torch.reshape(rewards, (-1, 1))
    q_eval = model(states).gather(1, actions) 
    q_next = target_model(next_states).detach()
    q_next[dones | truncated] = 0.0
    q_target = rewards + 0.99 * q_next.max(-1)[0].reshape(-1, 1)
    loss = loss_func(q_eval, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return learn_step_counter

def preprocess_observation(obs):
    return np.array(obs, dtype=np.float32)

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        q_values = model(state_tensor)
        return torch.argmax(q_values).item()
rewards = []

for episode in range(train_num):
    episode_start_time = time.time()  # Start timing the episode
    raw_state = env.reset()[0]
    state = preprocess_observation(raw_state)
    total_reward = 0
    learn_step_counter = 0
    now_length = 0
    done = False
    truncated = False

    while not done and not truncated:
        if now_length >= epsilode_length:
            break
        now_length += 1
        action = choose_action(state, epsilon_values[episode])
        raw_next_state, reward, done, truncated, info = env.step(action)
        next_state = preprocess_observation(raw_next_state)
        
        memory.append((state, action, reward, next_state, done, truncated))
        state = next_state
        total_reward += reward
        learn_step_counter = optimize_model(learn_step_counter)

    episode_end_time = time.time()  # End timing the episode
    rewards.append(total_reward)
    print("episode length:", now_length)
    print(f"Episode {episode + 1}: Total reward = {total_reward}. Episode duration: {episode_end_time - episode_start_time} seconds")

env.close()

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(range(1, train_num+1), rewards, marker='o', color='b', label='Total Reward per Episode')

coefficients = np.polyfit(range(1, train_num + 1), rewards, 1)
polynomial = np.poly1d(coefficients)
trendline = polynomial(range(1, train_num + 1))

# Adding the trend line to the plot
plt.plot(range(1, train_num + 1), trendline, color='r', label='Trend Line')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards as a Function of Training Episodes')
plt.legend()
plt.grid(True)
plt.savefig("./reward.png")
plt.show()