import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F # 四元数归一化

action_probs = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]
mu = torch.tensor(action_probs[0: 7])
sigma = torch.tensor(action_probs[7: 14])
# print(mu)
# print(sigma)
dist = Normal(loc=mu, scale=sigma)
sample = dist.sample()
log_prob = dist.log_prob(sample)
# print(dist, sample, log_prob)
x_np = sample.numpy()
sum1 = np.sum(x_np[3:7])
# print(sum1)
sample[3:7] = F.normalize(sample[3:7], p=2, dim=-1)
print(sample)
action = sample.numpy()
a=0
for i in range(3,7):
    a+=action[i]**2
    print(a)