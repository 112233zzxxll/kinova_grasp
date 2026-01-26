import mujoco
import mujoco.viewer
import numpy as np
import time
import nolo_tracker
import Kinematics as K
import cv2
import grasp_env as env
from datetime import datetime
import pickle
from pathlib import Path
import os
from ppo_agent import PPO
from ppo_agent import CNN
from torchvision import transforms
import torch
from PIL import Image
from mink.lie.so3 import SO3


# --------------- 基本配置 ---------------
model, data = env.init_env("mixed_model/scene.xml")
data.ctrl[:8] = [0,0,0,0,0,0,0,0]
n=1 # 索引
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for i in np.arange(-314,314,1):
            s=i/100
            data.ctrl[n] = s
            for k in range(10):
                mujoco.mj_step(model,data)
                viewer.sync()

