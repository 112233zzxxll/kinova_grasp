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
# 该脚本在control with nolo基础上做修改，进行数据采集
# 采集的数据包括：
# 1. 两个图像：img1 和 img2 （224*224*2）(observation)
# 2. 关节相对位姿：d_pos 和 d_so3 (action)
# 3. 瓶颈位置标记：由 True 和 False 构成的序列（skill）
# 4. 物体位姿和关节状态记录：记录 “瓶颈位置” 机械臂关节空间以及所有物体位姿


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

# --------------- 采集参数初始化 -------------------
actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator") # 夹爪执行器索引
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'right_driver_joint') # 关节索引
qpos_index = model.jnt_qposadr[joint_id] # 夹爪实际开度索引
# gripper_command = data.ctrl[actuator_id] # 控制指令
# gripper_actual = data.qpos[qpos_index] # 实际开度
step = 0
obs1 = []
obs2 = []
action = []
arm_state = []
object1 = []
object2 = []
skills = []
demo_dir = Path("demo_path")
demo_dir.mkdir(parents=True, exist_ok=True)


# --------------- 仿真 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        d_pos, d_so3, button = nolo_tracker.get_delta() # 获取相对位移
        if button > 0: #按下 trigger 开始动作
            ee_pos, ee_rot, target = K.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
            model.body("target").pos = ee_pos  # 立即生效
            quat = ee_rot.parameters()
            model.body("target").quat = quat
            if button > 10: # 夹爪闭合
                data.ctrl[7] = 255
            else:
                data.ctrl[7] = 0

            if button > 1000: # 环境重置 保存
                time.sleep(0.5)
                t = {}
                t["obs1"] = obs1
                t["obs2"] = obs2
                t["actions"] = action
                t["arm_state"] = arm_state
                t["object1"] = object1
                t["object2"] = object2
                t["skills"] = skills
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
                action = []
                arm_state = []
                object1 = []
                object2 = []
                skills = []
                print("缓存已清空")


            if step % 20 == 0:  # 每x步采一帧
                # --------------- 相机设置模块 ---------------
                renderer.update_scene(data, camera=cam1_id)
                img1 = renderer.render()
                renderer.update_scene(data, camera=cam2_id)
                img2 = renderer.render()
                # 显示离屏图像
                cv2.imshow("Top View", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
                cv2.imshow("Side View", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
                #---------------- 数据保存 -------------------
                obs1.append(img1)
                obs2.append(img2)
                action.append([d_pos, d_so3])
                arm_state.append(0)
                object1.append(0)
                object2.append(0)
                skills.append(False)
        step += 1

        result = K.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
        data.ctrl[:7] = result
        mujoco.mj_step(model, data)
        viewer.sync()
print("脚本终止")