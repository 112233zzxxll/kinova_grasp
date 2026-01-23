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
                 gae_lambda=0.95, # GAE 参数
                 epsilon=0.2 # 裁剪系数
                 ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon = epsilon

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer_actor = optim.Adam(self.policy.Actor.parameters(), lr=lr) # Actor 更新器
        self.optimizer_critic = optim.Adam(self.policy.Critic.parameters(), lr=lr) # Critic 更新器

        self.old_policy = ActorCritic(state_dim, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict()) # 复制策略

    def select_action(self, state):
        with torch.no_grad(): # 跳过梯度计算，防止反向传播
            # 注意这里的一些修改！！！！可能会报错
            action_probs, state_value = self.old_policy(state) # 旧策略采样

        # 重构输出的action_probs
        # 先按照概率采样，然后把四元数项归一化
        mu = torch.tensor(action_probs[0: 7])
        sigma = F.softplus(action_probs[7: 14]) + 1e-6
        dist = Normal(loc=mu, scale=sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        action[3:7] = F.normalize(action[3:7], p=2, dim=-1)

        return action.numpy(), log_prob.numpy(), state_value.numpy()
        # 根据观测，返回采样到的动作、对应的动作概率密度对数总和、状态价值

    def compute_gae(self, rewards, values):
        # 通过values获取nevt values
        next_values = np.zeros_like(values)
        for i in range(len(next_values)):
            if i == len(next_values) - 1:
                next_values[i] = 0
            else:
                next_values[i] = values[i+1]
    
        # 计算TD误差
        TD_errors = []
        for t in range(len(rewards)):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            TD_errors.append(delta)
        
        # 计算GAE优势
        advantages = []
        advantage = 0
        for l in reversed(range(len(TD_errors))):
            advantage = TD_errors[l] + self.gamma * self.gae_lambda * advantage
            advantages.insert(0, advantage)

        # 计算回报
        returns = []
        for r in range(len(rewards)):
            critic_return = advantages[r] + values[r]
            returns.append(critic_return)

        return advantages, returns
            
    def update(self, advantages, n_batchs, n_epoch, old_log_probs, states, actions, returns):
        # 先划分batch
        # 共 10 个 episode 的 rollout，每个 episode 包含 1000 步，总共 1000*10 组的参数随机打乱
        # 划分8个 batch，每个 batch 包含1250 个数据
        # n_epoch 是每组数据使用 n 次


        # 转为张量形式
        advantages = torch.tensor(advantages)
        old_log_probs = torch.tensor(old_log_probs)
        states = torch.tensor(states)
        actions  = torch.tensor(actions)
        returns = torch.tensor(returns)

        batch_size = len(advantages)//n_batchs
        for epoch in range(n_epoch): # 训练轮数

            T = len(advantages)
            indices = torch.randperm(T) # 索引随机打乱
            for i in range(0, len(advantages), batch_size):
                idx = indices[i: i+batch_size] # 随机索引
                # 针对每个 batch 进行如下操作
                # 重要性重采样
                new_action_probs, new_state_value = self.policy(states[idx])
                mu = new_action_probs[:, :7]
                sigma = F.softplus(new_action_probs[:, 7:]) + 1e-6
                dist = Normal(loc=mu, scale=sigma)
                new_log_probs = dist.log_prob(actions[idx]).sum(dim=1)

                # 计算概率比率
                ratios = torch.exp(new_log_probs - old_log_probs[idx])

                # Actor损失裁剪
                loss1 = ratios * advantages[idx]
                loss2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages[idx]
                Actor_loss = -torch.min(loss1, loss2).mean()

                # Actor 梯度更新
                self.optimizer_actor.zero_grad()
                Actor_loss.backward()
                # 梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(self.policy.Actor.parameters(), 0.5)
                self.optimizer_actor.step()

                critic_loss = nn.MSELoss()(new_state_value.squeeze(), returns[idx])
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.Critic.parameters(), 0.5)
                self.optimizer_critic.step()