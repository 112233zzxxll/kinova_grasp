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

# 该脚本用ppo训练kinova抓取拼接桁架
old_vel = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]] # 初值，用于计算与速度相关的奖励

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
img_tr.eval()

transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 训练主体
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        

        print("清空缓存")
        states_batch = []
        actions_batch = []
        action_probs_batch = []
        log_probs_batch = []
        values_batch = []
        rewards_batch = []
        advantages_batch = []
        returns_batch = []
        # rollout 10 次
        for episode in range(10):
            # 重置环境
            time.sleep(0.5)
            env.reset_target()
            ee_pos, ee_rot, target = env.reset_ee()
            data.ctrl[7] = 0
            states = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            for step in range(2000):
                print(step)
                mujoco.mj_step(model, data)
                viewer.sync()
                img1, img2 = env.get_obs() # 获取观测（state）
                img1_pil = Image.fromarray(img1.astype('uint8'))  # 确保是 uint8
                img2_pil = Image.fromarray(img2.astype('uint8'))
                img1 = transform(img1_pil).unsqueeze(0)
                img2 = transform(img2_pil).unsqueeze(0)
                with torch.no_grad():
                    img1 = img_tr(img1)
                    img2 = img_tr(img2)
                state = torch.cat([img1, img2], dim=1)[0]
                states.append(state)

                action, log_prob, value = net.select_action(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)

                # 动作拼接并执行
                data.ctrl[7] = action[0]
                data.ctrl[:7] = action[1:8]

                # 获取奖励
                reward, old_vel = env.get_reward(old_vel)
                rewards.append(reward)

            # 一次 rollout 完成
            print("一次 rollout 完成")
            advantages, returns = net.compute_gae(rewards, values)
            states_batch.append(states)
            actions_batch.append(actions)
            log_probs_batch.append(log_probs)
            values_batch.append(values)
            rewards_batch.append(rewards)
            advantages_batch.append(advantages)
            returns_batch.append(returns)

        
        # rollout 完成
        print("rollout 完成")




