import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F # 四元数归一化

reward = [1,2,3,4,5,6,7,8,9,0]
for t in reversed(range(len(reward))):
    print(reward[t])
for t in range(len(reward)):
    print(reward[t])