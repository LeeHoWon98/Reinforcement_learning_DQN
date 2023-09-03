import gymnasium
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from gymnasium_env_2 import CustomEnv_move
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
learning_rate = 0.0001
gamma = 0.98
buffer_limit = 50000
batch_size = 16

env = CustomEnv_move()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

#텐서보드 셋팅
f_dir = f'C:/ho_learing_3/gamma_{gamma}/lr_{learning_rate}_99/150000_E_bs_{batch_size}'
os.makedirs(f_dir, exist_ok = True)
writer = SummaryWriter(f_dir)

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    def __init__(self):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)
    

#DQN 모델
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def sample_action(self, obs, epsilon):
        obs=obs.cuda()
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 3)
        else:
            return out.argmax().item()
        
    

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)
        
        s = s.cuda()
        a = a.cuda()
        r = r.cuda()
        s_prime = s_prime.cuda()
        done_mask = done_mask.cuda()
        
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1).cuda()
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target).cuda()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
q = Qnet().cuda()
q_target = Qnet().cuda()
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()


print_interval = 1
print_intervals = 50
score = 0.0
optimizer = optim.Adam(q.parameters(), lr=learning_rate)

max_steps = 0

for n_epi in range(36000):
    if n_epi <= 26000:
        epsilon = max(0.01, 0.8 - (0.79 / 26000) * n_epi)  # Linear annealing from 0.8 to 0.01
    
    else:
        epsilon = 0.01
    
    find = False
    collision = False
    s = env.reset()

    while not find:
        print(s)
        a = q.sample_action(torch.from_numpy(s).float(), epsilon)
        s_prime, r, find, collision = env.step(a)
        done_mask = 0.0 if find or collision else 1.0
        memory.put((s, a, r / 100.0, s_prime, done_mask))
        s = s_prime

        score += r

        if find or collision:
            break

    if memory.size() > 6000:
        train(q, q_target, memory, optimizer)
        
    if n_epi % print_intervals == 0 and n_epi != 0:
        q_target.load_state_dict(q.state_dict())

    if n_epi % print_interval == 0 and n_epi != 0:
        #q_target.load_state_dict(q.state_dict())
        print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
            n_epi, score / print_interval, memory.size(), epsilon * 100))
        
        #tensorboard setting
        writer.add_scalar('Reward', score / print_interval, n_epi)
        writer.add_scalar('Epsilon', epsilon, n_epi)

        # Saving model parameters
        torch.save(q.state_dict(), "C:/test/ho_learing_3/q_model.pt")
        torch.save(q_target.state_dict(), "C:/test/ho_learing_3/q_target.pt")

        score = 0.0