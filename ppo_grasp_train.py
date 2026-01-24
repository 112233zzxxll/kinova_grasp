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
n_episode = 10 # 每轮 rollout 次数
n_step = 1000 # 每个 epoch 步长
n_train = 10 # 同一批 rollout 反复训练的次数
n_batch = 8
round = 5000 # 大轮

# 创建 checkpoint 目录
checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

# 训练主体
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        
        for i in range(round):
            print(f"=== 大轮 {i+1}/{round} ===")
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
            for episode in range(n_episode):
                # 重置环境
                time.sleep(0.5)
                env.reset_target()
                ee_pos, ee_rot, target = env.reset_ee()
                data.ctrl[7] = 0
                for i in range(1000):
                    result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                    data.ctrl[:7] = result
                    mujoco.mj_step(model, data)
                    viewer.sync()
                states = []
                actions = []
                log_probs = []
                values = []
                rewards = []
                print("环境已重置")
                for step in range(n_step):
                    # print(step)
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    # img1, img2 = env.get_obs() # 获取观测（state）
                    # 2. 异步获取图像（不会阻塞 mj_step）
                    try:
                        img1, img2 = async_renderer.render_async()
                    except Exception as e:
                        print(f"渲染失败: {e}")
                        continue
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
                    action_tensor = torch.from_numpy(action).float()
                    actions.append(action_tensor)
                    log_probs.append(log_prob)
                    values.append(value)

                    # 动作拼接并执行
                    data.ctrl[7] = action[0]
                    d_pos, d_so3 = action[1:4], action[4:8]
                    d_so3 = SO3(d_so3)
                    ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3)
                    result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                    data.ctrl[:7] = result
                    # # 另一种动作执行方式
                    # data.ctrl[7] = action[0]
                    # data.ctrl[:7] = action[1:8]

                    # 获取奖励
                    reward, old_vel = env.get_reward(old_vel)
                    rewards.append(reward)
                    print(f"\r{episode+1}/{n_episode} ||| {step+1} ||| {sum(rewards):.2f}", end="", flush=True)

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




