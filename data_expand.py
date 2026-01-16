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
每次运行前需要读取单条.pkl文件, 扩充后将生成5*5条新数据, 预计总共20条
数据格式: demo_path/.pkl
    obs1  手臂相机
    obs2  正面相机
    actions  输入动作
    state_dict 瓶颈状态
    object  被抓物体位姿
    skills  瓶颈位置索引
    targets  求解状态
    ee  末端位姿
"""
# 打开并加载 .pkl 文件
demo_dir = Path("demo_path")
pkl_files = sorted(demo_dir.glob("*.pkl"))
first_file = pkl_files[0]
with open(first_file, "rb") as f:
    data0 = pickle.load(f)

obs1_origin = data0["obs1"]
obs2_origin = data0["obs2"]
action_origin = data0["actions"]
state_dict_origin = data0["state_dict"]
object_origin = data0["object"]
skills_origin = data0["skills"]
targets_origin = data0["targets"]
ee_origin = data0["ee"]

# --------------- 基本配置 ---------------
model, data = env.init_env("mixed_model/scene.xml")
env.reset_target()

# --------------- 启用nolo ---------------
nolo_tracker.func()
button = 0
# --------------- 初始化target ---------------
ee_pos, ee_rot, target = env.reset_ee()

# ----------------- 初始化求解模型 ----------------
configuration, tasks, end_effector_task, solver, limits, model0, data0 = K.init_model() # 所有求解中间变量打包

# --------------- 相机预设置 ---------------
cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
renderer = mujoco.Renderer(model, height=224, width=224)

# --------------- 键盘监听 ------------------
user_input = None
def key_callback(keycode):
    global user_input
    if keycode == ord('Y'):
        user_input = "save"
    elif keycode == ord('U'):
        user_input = "abandon"
    print(f"Key pressed: {chr(keycode)}")

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

step = 0
obs1 = [] # 手臂相机
obs2 = [] # 正面相机
action = [] # 输入动作
state_dict = [] # 瓶颈状态
object = [] # 目标位置
demo_dir = Path("expanded_path")
demo_dir.mkdir(parents=True, exist_ok=True)
change = False # 是否切换到下一条进行采集

# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running(): 
        for i in range(len(skills_origin)): # 瓶颈位置
            for j in range(8): # 每个瓶颈位置扩充 8 条数据

                restore_state(data, state_dict_origin[i])
                model.body_pos[object_body_id1] = object_origin[i] # 重置对应瓶颈位置

                # 这里把隐含的 target 算出来
                restore_state(data0, targets_origin[i])
                [ee_rot, ee_pos, target, result] = ee_origin[i]
                print(ee_origin)
                data.ctrl[:7] = result
                data.ctrl[7] = 255
                model.body("target").pos = ee_pos  
                quat = ee_rot.parameters()
                model.body("target").quat = quat
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.5)
                print("环境重置完毕")

                while not change: # 数据采集
                    d_pos, d_so3, button = nolo_tracker.get_delta() # 获取相对位移
                    ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
                    result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
                    data.ctrl[:7] = result
                #     # target 也要保存，否则机械臂运行就闪现？
                    if button > 0: #按下 trigger 开始动作
                        ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
                        model.body("target").pos = ee_pos  # 立即生效
                        quat = ee_rot.parameters()
                        model.body("target").quat = quat
                        if button > 10: # 夹爪闭合
                            data.ctrl[7] = 255
                        else:
                            data.ctrl[7] = 0
                    mujoco.mj_step(model, data)
                    viewer.sync()
