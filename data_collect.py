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
# 该脚本在control with nolo基础上做修改，进行数据采集
# 采集的数据包括：
# 1. 两个图像：img1 和 img2 （224*224*2）(observation)
# 2. 关节相对位姿：d_pos 和 d_so3 (actions)
# 3. 瓶颈位置标记：瓶颈位置索引（skill）
# 4. 物体位姿和关节状态记录：记录 “瓶颈位置” 机械臂关节空间以及所有物体位姿
""" 
数据格式：.pkl
    obs1  手臂相机
    obs2  正面相机
    actions  输入动作
    state_dict 瓶颈状态
    object  被抓物体位姿
    skills  瓶颈位置索引
    ee 末端位姿
"""


# --------------- 基本配置 ---------------
model, data = env.init_env("mixed_model/scene.xml")
env.reset_target()

# --------------- 启用nolo ---------------
nolo_tracker.func()

# --------------- 初始化target ---------------
ee_pos, ee_rot, target = env.reset_ee()

# ----------------- 初始化求解模型 ----------------
configuration, tasks, end_effector_task, solver, limits, model0, data0 = K.init_model() # 所有求解中间变量打包

# --------------- 相机预设置 ---------------
cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
renderer = mujoco.Renderer(model, height=112, width=112)

# --------------- 键盘监听 ------------------
user_input = None
def key_callback(keycode):
    global user_input
    if keycode == ord('Y'):
        user_input = "save"
    elif keycode == ord('U'):
        user_input = "abandon"
    print(f"Key pressed: {chr(keycode)}")

# --------------- 环境重构函数定义 -------------------
def save_state(data): # 保存状态
    return {
        'time': data.time,
        'qpos': data.qpos.copy(),
        'qvel': data.qvel.copy(),
        'act': data.act.copy() if data.act.size > 0 else None,
        'mocap_pos': data.mocap_pos.copy(),
        'mocap_quat': data.mocap_quat.copy(),
    }

def restore_state(data, state_dict): # 加载状态
    data.time = state_dict['time']
    data.qpos[:] = state_dict['qpos']
    data.qvel[:] = state_dict['qvel']
    if state_dict['act'] is not None:
        data.act[:] = state_dict['act']
    data.mocap_pos[:] = state_dict['mocap_pos']
    data.mocap_quat[:] = state_dict['mocap_quat']

object_body_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "assemble") # 修改默认位置
# model.body_pos[object_body_id1] = random_pos # 默认位置修改

step = 0
obs1 = [] # 手臂相机
obs2 = [] # 正面相机
actions = [] # 输入动作
state_dict = [] # 瓶颈状态
object = [] # 目标位置
skills = [] # 瓶颈位置索引
ee = [] # 瓶颈状态末端位姿
demo_dir = Path("demo_path")
demo_dir.mkdir(parents=True, exist_ok=True)

# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():

        d_pos, d_so3, button = nolo_tracker.get_delta() # 获取相对位移
        result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
        data.ctrl[:7] = result
        mujoco.mj_step(model, data)
        viewer.sync()
        if button > 0: #按下 trigger 开始动作
            ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
            model.body("target").pos = ee_pos  # 立即生效
            quat = ee_rot.parameters()
            model.body("target").quat = quat
            if button > 10: # 夹爪闭合
                data.ctrl[7] = 255
            else:
                data.ctrl[7] = 0

            if step % 20 == 0 or button > 1000:  # 每x步采一帧
                # --------------- 相机设置模块 ---------------
                renderer.update_scene(data, camera=cam1_id)
                img1 = renderer.render()
                renderer.update_scene(data, camera=cam2_id)
                img2 = renderer.render()
                # 显示离屏图像
                cv2.imshow("Top View", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                cv2.imshow("Side View", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                skill = len(obs1) + 1 # 瓶颈状态索引
                #---------------- 数据保存 -------------------
                obs1.append(img1)
                obs2.append(img2)
                actions.append([d_pos, d_so3])
                if button > 1000: # 记录瓶颈状态
                    time.sleep(1)
                    state = save_state(data)
                    state_dict.append(state)
                    skills.append(skill)
                    object.append(model.body_pos[object_body_id1])
                    ee0 = K.get_ee()
                    ee.append(ee0)
                    print("瓶颈状态已记录：", skills)
            step += 1

        if user_input == "save": # 环境重置 保存
            time.sleep(0.5)
            t = {}
            t["obs1"] = obs1
            t["obs2"] = obs2
            t["actions"] = actions
            t["state_dict"] = state_dict
            t["object"] = object
            t["skills"] = skills
            t["ee"] = ee
            path = demo_dir / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.pkl"
            with open(path, "wb") as f:
                pickle.dump(t, f)
            print(f"Data saved at {path}")
            print("数据已保存")
            time.sleep(0.5)
            env.reset_target()
            ee_pos, ee_rot, target = env.reset_ee()
            data.ctrl[7] = 0
            step = 0
            obs1 = []
            obs2 = []
            actions = []
            state_dict = []
            object = []
            skills = []
            ee = []
            print("缓存已清空")
            user_input = None
        elif user_input == "abandon": # 环境重置 不保存
            env.reset_target()
            ee_pos, ee_rot, target = env.reset_ee()
            data.ctrl[7] = 0
            step = 0
            obs1 = []
            obs2 = []
            actions = []
            state_dict = []
            object = []
            skills = []
            ee = []
            print("正在清理......")
            user_input = None
            time.sleep(1)
            print("缓存已清空,数据已抛弃")

print("脚本终止，重启终端以重新运行程序")