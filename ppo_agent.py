import torch
import torch.nn as nn
from torch.distributions import Normal # 正态分布采样
import torch.nn.functional as F # 四元数归一化
import torch.optim as optim
import numpy as np

class CNN(nn.Module): # 图像卷积，不参与参数更新，卷积前记得进行图像预处理，要求输入112*112的img
    def __init__(self):
        super().__init__()
        # 共享的图片处理层 输入为 112 * 112 这里只处理了一个图像，输出为256
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2), # 32 112 112 
            nn.ReLU(),
            nn.MaxPool2d(2), # 32 56 56
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2), # 64 56 56
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 64 4 4
            nn.Flatten() # 展平
        )
        self.fc = nn.Linear(64*4*4, 256)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim): # hidden_dim = 32
        super().__init__()
        # Actor 网络
        # 输出动作的正态分布，数据结构为[mu1,mu2,...,mu7,sigma1, sigma2,...,sigma7]
        # 前三个是dx,dy,dz,后四个是d_so3四元数
        self.Actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*action_dim), 
        )

        # Critic网络
        # 增加隐藏层，增强数据表达能力
        self.Critic = nn.Sequential(
            nn.Linear(state_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Linear(4*hidden_dim, 1), 
        )

    def forward(self, x):
        action_probs = self.Actor(x)
        state_value = self.Critic(x)
        return action_probs, state_value
    
class PPO:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 lr=1e-4, 
                 gamma=0.99, # 折扣因子
                 gae_lambda=0.95 # GAE 参数
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimaizer_actor = optim.Adam(self.policy.Actor.parameters(), lr=lr) # Actor 更新器
        self.optimaizer_critic = optim.Adam(self.policy.Critic.parameters(), lr=lr) # Critic 更新器

        self.old_policy = ActorCritic(state_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict()) # 复制策略

    def select_action(self, state):
        with torch.no_grad(): # 跳过梯度计算，防止反向传播
            # 注意这里的一些修改！！！！可能会报错
            action_probs, state_value = self.old_policy(state) # 旧策略采样

        # 重构输出的action_probs
        # 先按照概率采样，然后把四元数项归一化
        mu = torch.tensor(action_probs[0: 7])
        sigma = torch.tensor(action_probs[7: 14])
        dist = Normal(loc=mu, scale=sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action[3:7] = F.normalize(action[3:7], p=2, dim=-1)

        return action.numpy(), log_prob.numpy(), state_value.numpy()
        # 根据观测，返回采样到的动作、对应的动作概率密度对数、状态价值

    def compute_gae(self, rewards, values):
        # 通过values获取nevt values
        next_values = values
        next_values.pop(0)
        next_values.append(values[-1]) 

        # 计算TD误差
        TD_errors = []
        for t in range(len(rewards)):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            TD_errors.append(delta)
        
        # 计算GAE优势
        advantages = []
        advantage = 0
        for l in reversed(range(len(TD_errors))):
            advantage += TD_errors[l] * self.gae_lambda * self.gamma
            advantages.insert(0, advantage)

        # 计算回报
        returns = []
        for r in range(len(rewards)):
            critic_return = advantages[r] + rewards[r]
            returns.append(critic_return)

        return advantages, returns
            