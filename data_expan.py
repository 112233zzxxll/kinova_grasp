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

"""
本程序用于对采集到的数据进行重组扩充
数据格式: demo_path/.pkl
    obs1  手臂相机
    obs2  正面相机
    action  输入动作
    arm_state  手臂状态(重建场景使用)
    object1  被抓物体位姿
    object2  装配目标位置(无旋转)
    skills  瓶颈位置索引

"""
# 打开并加载 .pkl 文件
path = "demo_path/2026-01-15T19-57-09.pkl"
with open(path, "rb") as f:  # 注意是 "rb"（二进制读取）
    data = pickle.load(f)

# 现在 data 就是你保存时的原始对象（字典、列表等）
print(data.keys())  # 如果是字典
print(data["arm_state"])  # 示例




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
renderer = mujoco.Renderer(model, height=224, width=224)

# --------------- 采集参数初始化（用于环境重构） -------------------
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator") # 夹爪执行器索引
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_driver_joint') # 关节索引
qpos_index = model.jnt_qposadr[joint_id] # 夹爪实际开度索引
# data.ctrl[actuator_id] # 控制指令
# data.qpos[qpos_index] # 实际开度

object1_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "six_joint")
object1_qpos_adr = model.jnt_qposadr[object1_joint_id]
# data.qpos[object1_qpos_adr:object1_qpos_adr + 3] # 对应 pos
# data.qpos[object1_qpos_adr + 3:object1_qpos_adr + 7] # 对应 quat

object2_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "assemble") #修改默认位置
# model.body_pos[object2_body_id]  # 位置

# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
print("脚本终止")