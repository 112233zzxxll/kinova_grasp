import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F # 四元数归一化

rewards = [1,2,3,4,5,6,7,8,9,0]
values = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
def compute_gae(rewards, values):
    # 通过values获取nevt values
    next_values = values
    next_values.pop(0)
    next_values.append(values[-1]) 

    # 计算TD误差
    TD_errors = []
    for t in range(len(rewards)):
        delta = rewards[t] + 0.99 * next_values[t] - values[t]
        TD_errors.append(delta)
    
    # 计算GAE优势
    advantages = []
    advantage = 0
    for l in reversed(range(len(TD_errors))):
        advantage += TD_errors[l] * 0.95 * 0.99
        advantages.insert(0, advantage)

    # 计算回报
    returns = []
    for r in range(len(rewards)):
        critic_return = advantages[r] + rewards[r]
        returns.append(critic_return)

    return advantages, returns

advantages, returns = compute_gae(rewards, values)
print(advantages)
print(returns)