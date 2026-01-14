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

# --------------- 相机预设置 ---------------
cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
renderer = mujoco.Renderer(model, height=224, width=224)

# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():

        # --------------- 相机设置模块 ---------------
        if step % 100 == 0:  # 每100步采一帧
            renderer.update_scene(data, camera=cam1_id)
            img1 = renderer.render()
            renderer.update_scene(data, camera=cam2_id)
            img2 = renderer.render()
            # 显示离屏图像
            cv2.imshow("Top View", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
            cv2.imshow("Side View", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        d_pos, d_so3, button = nolo_tracker.get_delta() # 获取相对位移
        if button > 0:
            step += 1
            print(step)
            ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
            model.body("target").pos = ee_pos  # 立即生效
            quat = ee_rot.parameters()
            model.body("target").quat = quat
            if button > 10:
                data.ctrl[7] = 255
            else:
                data.ctrl[7] = 0
            if button > 1000: # 环境重置
                time.sleep(0.5)
                env.reset_target()
                ee_pos, ee_rot, target = env.reset_ee()
                data.ctrl[7] = 0
                step = 0
        result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
        data.ctrl[:7] = result
        mujoco.mj_step(model, data)
        viewer.sync()
print("脚本终止")