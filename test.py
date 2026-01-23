import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F # 四元数归一化
import torch.nn as nn

values = [1,2,3,4,5]
next_values = values
next_values.pop(0)
next_values.append(values[-1]) 
print(values)
print(next_values)