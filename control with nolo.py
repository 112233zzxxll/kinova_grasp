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


# --------------- 基本配置 ---------------
model, data = env.init_env("mixed_model/scene.xml")
env.reset_target()

# --------------- 启用nolo ---------------
nolo_tracker.func()

# --------------- 初始化target ---------------
ee_pos, ee_rot, target = env.reset_ee()

# ----------------- 初始化求解模型 ----------------
configuration, tasks, end_effector_task, solver, limits, model0, data0 = K.init_model() # 所有求解中间变量打包

# # --------------- 相机预设置 ---------------


# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        # # --------------- 相机设置模块 ---------------


        d_pos, d_so3, button = nolo_tracker.get_delta() # 获取相对位移
        if button > 0:
            ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
            model.body("target").pos = ee_pos  # 立即生效
            quat = ee_rot.parameters()
            model.body("target").quat = quat
            if button > 10:
                data.ctrl[7] = 255
            else:
                data.ctrl[7] = 0
            if button > 1000:
                time.sleep(0.3)
                env.reset_target()
                ee_pos, ee_rot, target = env.reset_ee()
                data.ctrl[7] = 0
        result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
        data.ctrl[:7] = result
        mujoco.mj_step(model, data)
        viewer.sync()