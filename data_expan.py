import pickle
import mujoco
import mujoco.viewer
import numpy as np
import time
import nolo_tracker
import mink
import Kinematics as K
import glfw
import cv2
import grasp_env as env
from pathlib import Path

"""
本程序用于对采集到的数据进行重组扩充
数据格式: demo_path/.pkl
    obs1  手臂相机
    obs2  正面相机
    action  输入动作
    state_dict 瓶颈状态
    object  被抓物体位姿
    skills  瓶颈位置索引

"""
# 打开并加载 .pkl 文件
demo_dir = Path("demo_path")
pkl_files = sorted(demo_dir.glob("*.pkl"))
first_file = pkl_files[0]
with open(first_file, "rb") as f:
    data0 = pickle.load(f)

state_dict = data0["state_dict"]
object = data0["object"]




# --------------- 基本配置 ---------------
model, data = env.init_env("mixed_model/scene.xml")
env.reset_target()

# --------------- 启用nolo ---------------
# nolo_tracker.func()

# --------------- 初始化target ---------------
ee_pos, ee_rot, target = env.reset_ee()

# ----------------- 初始化求解模型 ----------------
configuration, tasks, end_effector_task, solver, limits, model0, data0 = K.init_model() # 所有求解中间变量打包

# --------------- 相机预设置 ---------------
cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
renderer = mujoco.Renderer(model, height=224, width=224)


# --------------- 环境重置 ---------------
def save_state(data):
    return {
        'time': data.time,
        'qpos': data.qpos.copy(),
        'qvel': data.qvel.copy(),
        'act': data.act.copy() if data.act.size > 0 else None,
        'mocap_pos': data.mocap_pos.copy(),
        'mocap_quat': data.mocap_quat.copy(),
    }
def restore_state(data, state_dict):
    data.time = state_dict['time']
    data.qpos[:] = state_dict['qpos']
    data.qvel[:] = state_dict['qvel']
    if state_dict['act'] is not None:
        data.act[:] = state_dict['act']
    data.mocap_pos[:] = state_dict['mocap_pos']
    data.mocap_quat[:] = state_dict['mocap_quat']

object_body_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "assemble") # 修改默认位置

# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        restore_state(data, state_dict[1])
        model.body_pos[object_body_id1] = object[0]
        mujoco.mj_step(model, data)
        viewer.sync()
print("脚本终止")