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

# 该脚本用ppo训练kinova抓取拼接桁架
old_vel = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]

# --------------- 基本配置 ---------------
model, data = env.init_env("mixed_model/scene.xml")
env.reset_target()

# --------------- 初始化target ---------------
ee_pos, ee_rot, target = env.reset_ee()

# ----------------- 初始化求解模型 ----------------
configuration, tasks, end_effector_task, solver, limits, model0, data0 = K.init_model() # 所有求解中间变量打包

# --------------- 相机预设置 ---------------
cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
renderer = mujoco.Renderer(model, height=112, width=112)

# ---------------- 初始化神经网络 --------------
net = PPO(256, 8, 32)
img_tr = CNN()

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 训练主体
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # rollout 10 次
        for episode in range(10):
            # 重置环境
            time.sleep(0.5)
            env.reset_target()
            ee_pos, ee_rot, target = env.reset_ee()
            data.ctrl[7] = 0
            for step in range(2000):
                img1, img2 = env.get_obs()
                img1 = transform(img1)
                img2 = transform(img2)
                img1 = img_tr(img1)
                img2 = img_tr(img2)
                