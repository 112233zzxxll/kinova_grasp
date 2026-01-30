import mujoco
import mujoco.viewer
import numpy as np
import time
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
from camera_opr import AsyncRenderer


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
async_renderer = AsyncRenderer(
    model, 
    data, 
    cameras=[cam1_id, cam2_id],
    height=112,
    width=112
)

# ---------------- 初始化神经网络 --------------
net = PPO(256, 8, 32)
img_tr = CNN()
img_tr.eval()

transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

############################################训练参数####################################################
# n_episode = 1 # 每轮 rollout 次数
# n_step = 50 # 每个 epoch 步长
# n_train = 5 # 同一批 rollout 反复训练的次数
# n_batch = 5
# round = 50000 # 大轮
# n = 100 # 每隔n个step执行一次动作

n_episode = 10 # rollout 次数
n_step = 1000 # 每个 epoch 步长
n_train = 10 # 同一批 rollout 反复训练的次数
n_batch = 8
n = 200 # 每隔n个执行一次动作
########################################################################################################

# 创建 checkpoint 目录
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# 训练主体
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
            
            for episode in range(n_episode):
                # 重置环境
                time.sleep(0.1)
                ee_pos, ee_rot, target = env.reset_ee()
                data.ctrl[7] = 0
                for i in range(1000):
                    result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                    data.ctrl[:7] = result
                    mujoco.mj_step(model, data)
                    viewer.sync()
                env.reset_target()
                time.sleep(0.1)
                states = []
                next_states = []
                actions = []
                log_probs = []
                values = []
                next_values = []
                rewards = []
                dones = []
                print("环境已重置")


                # rollout
                img1, img2 = env.get_obs() # 获取观测（state）
                # 视觉编码
                with torch.no_grad():
                    img1 = img_tr(img1.unsqueeze(0)).numpy()[0]
                    img2 = img_tr(img2.unsqueeze(0)).numpy()[0]
                state = []
                state.extend(img1)
                state.extend(img2)
                i = 0 # 记录一局是否结束
                for step in range(n_step):
                    print(f"\r{step}", end="", flush=False)
                    # mujoco.mj_step(model, data)
                    # viewer.sync()#####################################记得后面加
                    states.append(state)
                    # print(state)
                    action, log_prob, value = net.select_action([state])
                    # print(action)
                    actions.append(action[0])
                    # print(action, log_prob, value)
                    log_probs.append(log_prob)
                    values.append(value)

                    # # 动作执行，作为末端位姿的差值输出，记得打开四元数归一化
                    # data.ctrl[7] = action[0]
                    # d_pos, d_so3 = action[1:4], action[4:8]
                    # d_so3 = SO3(d_so3)
                    # ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3)
                    # result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                    # data.ctrl[:7] = result

                    # # 另一种动作执行方式，直接映射为关节角，记得关闭四元数归一化
                    # data.ctrl[7] = action[0]
                    # result = action[1:8]
                    # data.ctrl[:7] = result

                    # # 还有一种方式，直接作为末端位姿， 记得打开四元数归一化
                    # ee_pos, ee_rot = action[1:4], action[4:8]
                    # ee_rot = SO3(ee_rot)
                    # d_pos = [0,0,0]
                    # d_so3 = np.array([1,0,0,0])
                    # d_so3 = SO3(d_so3)
                    # ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3)
                    # result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                    # data.ctrl[:7] = result
                    # data.ctrl[7] = action[0]

                    # 最后一种动作执行方式，关节角差值，记得关闭四元数归一化,
                    # 问题是关节过度扭转
                    data.ctrl[7] = action[0][0]
                    result += action[0][1:8]
                    data.ctrl[:7] = result

                    model.body("target").pos = ee_pos  # 立即生效
                    quat = ee_rot.parameters()
                    model.body("target").quat = quat

                    # 获取奖励
                    reward, old_vel, done = env.get_reward(old_vel, i)
                    rewards.append(reward)

                    if done == 1: # 结束
                        next_values.append(0.0)
                         # 重置环境
                        time.sleep(0.1)
                        ee_pos, ee_rot, target = env.reset_ee()
                        data.ctrl[7] = 0
                        for i in range(1000):
                            result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                            data.ctrl[:7] = result
                            mujoco.mj_step(model, data)
                            viewer.sync()
                        env.reset_target()
                        time.sleep(0.1)
                        img1, img2 = env.get_obs() # 获取观测（state）
                        i = -1
                        # 视觉编码
                        with torch.no_grad():
                            img1 = img_tr(img1.unsqueeze(0)).numpy()[0]
                            img2 = img_tr(img2.unsqueeze(0)).numpy()[0]
                        state = []
                        state.extend(img1)
                        state.extend(img2)

                    else:
                        # 前进获取 next_value
                        img1, img2 = env.get_obs() # 获取观测（state）
                        # 视觉编码
                        with torch.no_grad():
                            img1 = img_tr(img1.unsqueeze(0)).numpy()[0]
                            img2 = img_tr(img2.unsqueeze(0)).numpy()[0]
                        next_state = []
                        next_state.extend(img1)
                        next_state.extend(img2)
                        # print(state)
                        action, log_prob, next_value = net.select_action([next_state])
                        next_values.append(next_value)
                        state = next_state

                    i += 1  
                    # # 保证策略输出正常
                    # for i in range(20):
                    #     # data.ctrl[:7] = result
                    #     # data.ctrl[7] = action[0] 
                    #     mujoco.mj_step(model, data)
                    #     viewer.sync()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                # 一次 rollout 完成
                print("一次 rollout 完成")

            
            # rollout 完成
            print("rollout 完成")
            states_batch = torch.cat([torch.stack(ep) for ep in states_batch], dim=0)        # [N, D]
            actions_batch = torch.cat([torch.stack(ep) for ep in actions_batch], dim=0)      # [N, A]
            log_probs_batch = torch.tensor([log for ep in log_probs_batch for log in ep])
            advantages_batch = [a for ep in advantages_batch for a in ep]
            returns_batch = [r for ep in returns_batch for r in ep]
            net.update(advantages_batch, n_batch, n_train, log_probs_batch, states_batch, actions_batch, returns_batch)
            print("更新完成")
            checkpoint_path = checkpoint_dir / f"ppo_kinova_grasp_round_{i+1}.pth"
            torch.save({
                'epoch': i + 1,
                'policy_state_dict': net.policy.state_dict(),
                'actor_optimizer_state_dict': net.optimizer_actor.state_dict(),
                'critic_optimizer_state_dict': net.optimizer_critic.state_dict(),
                'cnn_state_dict': img_tr.state_dict(),
                'epsilon': net.epsilon,
                'gamma': net.gamma,
                'gae_lambda': net.gae_lambda,
            }, checkpoint_path)
            print(f"✅ 检查点已保存: {checkpoint_path}")




