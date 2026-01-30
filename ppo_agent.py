import torch
import torch.nn as nn
from torch.distributions import Normal # 正态分布采样
import torch.nn.functional as F # 四元数归一化、
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
        self.fc = nn.Linear(64*4*4, 128)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim): # hidden_dim = 32
        super().__init__()
        self.action_dim = action_dim
        # Actor 网络
        # 输出动作的正态分布，数据结构为[mu1,mu2,...,mu7,mu8,sigma1, sigma2,...,sigma7, sigma8]
        # 第一个是夹爪
        # 而后三个是dx,dy,dz,最后四个是d_so3四元数
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
            nn.Linear(state_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear(2*hidden_dim, 1), 
        )

    def forward(self, x):
        action_probs = self.Actor(x)
        state_value = self.Critic(x)
        # 拆分 mean 和 log_std
        mean_raw = action_probs[:,:self.action_dim]  # 动作均值
        std = action_probs[:,self.action_dim:]   # 动作标准差

        # 首先对均值进行调整：
        # 对 mean 做 tanh 归一化到 [-1, 1]
        mean_tanh = torch.tanh(mean_raw)
        # ✅ 安全地缩放不同维度（全部 out-of-place）
        # 夹爪维度 (dim=0): [-1,1] -> [0, 255](500)
        gripper = (mean_tanh[:,0:1] + 1) / 2 * 500  
        # 其他自由度（可以是位移和四元数，也可以是七个关节）
        freedom = mean_tanh[:, 1:self.action_dim] * 0.02

        # 然后对方差进行调整
        std = (torch.tanh(std)+1)/2 # 转换到0~1之间
        std = std * 0.01 + 1e-6 # 动作采样稳定性控制 ######################################################################

        # 拼接：1D 张量用 dim=0
        mean_scaled = torch.cat([gripper, freedom], dim=1)
        action_probs_new = torch.cat([mean_scaled, std], dim=1)

        return action_probs_new, state_value # action_probs：前八维是均值，后八维是方差，如果想要四元数，需要在select_action归一化
    
class PPO:
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 hidden_dim,
                 lr=0.0004, 
                 gamma=0.99, # 折扣因子
                 gae_lambda=0.95, # GAE 参数
                 epsilon=0.2 # 裁剪系数
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon

        self.policy = ActorCritic(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer_actor = optim.Adam(self.policy.Actor.parameters(), lr=lr) # Actor 更新器
        self.optimizer_critic = optim.Adam(self.policy.Critic.parameters(), lr=lr) # Critic 更新器

    def select_action(self, state): # 输入 numpy 数组
        state = torch.tensor(state, dtype=torch.float32) # 先转 torch 张量
        with torch.no_grad(): # 跳过梯度计算，防止反向传播
            # 注意这里的一些修改！！！！可能会报错
            action_probs, state_value = self.policy(state) # 策略采样
        # 按照概率采样，（把四元数项归一化）
        mu = action_probs[:, 0: self.action_dim]
        sigma = action_probs[:, self.action_dim: 2*self.action_dim]
        dist = Normal(loc=mu, scale=sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        # action[4:8] = F.normalize(action[4:8], p=2, dim=-1) # 四元数归一化
        return action.numpy(), log_prob.item(), state_value.item() 
        # 根据观测，返回采样到的动作、对应的动作概率密度对数总和、状态价值

    def compute_gae(self, rewards, values, next_values, dones):
        # 针对一次 rollout 数据进行计算
        # 输入 numpy 数组
        advantages = []
        advantage = 0
        returns = []
        for i in reversed(range(len(dones))):
            TD_erro = rewards[i] + self.gamma * next_values[i] - values[i]
            advantage = TD_erro + self.gamma * self.gae_lambda * advantage * (1-dones[i])
            c_return = advantage + values[i]
            advantages.insert(0, advantage)
            returns.insert(0, c_return)
        return advantages, returns
            
    def update(self, n_epoch, n_batch, advantages, returns, old_log_probs, states, actions):
        # n_epoch 是数据集被遍历的次数
        # n_batch 是每次遍历时分包个数
        # old_log_probs, states, actions用于重要性采样
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        batch_size = len(advantages)//n_batch
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 标准化

        for epoch in range(n_epoch): 
            # 首先划分 batch
            T = len(advantages)
            indices = torch.randperm(T)
            for i in range(0, T, batch_size):
                idx = indices[i: i+batch_size]

                # 在每个 batch 下进行重要性采样
                new_action_probs, new_state_value = self.policy(states[idx])
                # 注意这里的一些修改！！！！可能会报错
                # 按照概率采样，（把四元数项归一化）
                mu = new_action_probs[:, 0: self.action_dim]
                sigma = new_action_probs[:, self.action_dim: 2*self.action_dim]
                dist = Normal(loc=mu, scale=sigma)
                new_log_probs = dist.log_prob(actions[idx])

                # 计算概率比率
                ratios = torch.exp(new_log_probs.sum(dim=1) - old_log_probs[idx])
                # Actor损失裁剪
                loss1 = ratios * advantages[idx]
                loss2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages[idx]
                Actor_loss = -torch.min(loss1, loss2).mean()
                # 更新 Actor
                self.optimizer_actor.zero_grad()
                Actor_loss.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # 更新 Critic
                critic_loss = nn.MSELoss()(new_state_value.squeeze(-1), returns[idx])
                # print(critic_loss)
                # print(new_state_value.squeeze(),"|||",  returns[idx])
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), 0.5)
                self.optimizer_critic.step()