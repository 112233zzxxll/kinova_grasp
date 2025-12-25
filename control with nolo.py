import mujoco
import mujoco.viewer
import numpy as np
import random
import time
import nolo_tracker
import mink
import mink_use

# --------------- 基本配置 ---------------
model = mujoco.MjModel.from_xml_path('mixed_model/scene.xml')
data  = mujoco.MjData(model)


pos1 = [0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633, 0] # 夹爪向下
pos2 = [0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633, 0] # 夹爪水平
pos3 = [0, -0.382, 3.14159265, -1.75, 0, -1.43, 1.57079633, 0] # 夹爪斜向下
data.ctrl[:] = pos3

dt = model.opt.timestep
duration = 2
steps = int(duration / dt)          # 2 秒对应的步数


# -------------- 待抓取模型随机生成 -------------
"""工作空间"""
workspace = {
            'x': [-0.6, -0.2], # -0.6 离底座最近
            'y': [1.0, 1.4], # 1.4 为中间
            'z': [1.35, 1.35] # 固定生成高度
            }
"""在工作空间内生成随机位置"""
x = np.random.uniform(*workspace['x'])
y = np.random.uniform(*workspace['y'])
z = np.random.uniform(*workspace['z'])
random_pos = np.array([x, y, z])
print(random_pos)
"""获取物体id"""
object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "six")
object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "six_joint")
object_qpos_adr = model.jnt_qposadr[object_joint_id]
"""赋予位置"""
model.body_pos[object_body_id] = random_pos
data.qpos[object_qpos_adr:object_qpos_adr + 3] = random_pos


# 启用nolo,初始化原始位姿：
nolo_tracker.func()
ee_pos = np.array([0,0,0])
ee_rot = np.array([1,0,0,0], dtype=np.float64)
ee_rot = mink.SO3(ee_rot)  # 构造 SO3 群元

ee_pos = np.array([0.3, 0, 0.5])
ee_rot = np.array([[0.5,0.866,0], [0.866,-0.5,0], [0,0,-1]])
ee_rot = mink.SO3.from_matrix(ee_rot)
target = mink.SE3.from_rotation_and_translation(ee_rot, ee_pos) # 对准，这个 target 后续切换成物体位位姿估计

# 初始化求解模型
configuration, tasks, end_effector_task, solver, limits, model0, data0 = mink_use.init_model() # 所有求解中间变量打包

# --------------- 线性流程 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        d_pos, d_so3, button = nolo_tracker.get_delta() # 获取相对位移
        if button > 0:
            ee_pos, ee_rot, target = mink_use.get_target(ee_pos, ee_rot, d_pos, d_so3) # 末端位姿
            model.body("target").pos = ee_pos  # 立即生效
            quat = ee_rot.parameters()
            model.body("target").quat = quat
            if button > 10:
                data.ctrl[7] = 255
            else:
                data.ctrl[7] = 0
        result = mink_use.solve(configuration, tasks, end_effector_task, solver, limits, model0, data0, target)
        data.ctrl[:7] = result
        mujoco.mj_step(model, data)
        viewer.sync()