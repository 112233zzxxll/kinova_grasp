import mujoco
import mujoco.viewer
import numpy as np
import random
import time

# --------------- 基本配置 ---------------
model = mujoco.MjModel.from_xml_path('mixed_model/scene.xml')
data  = mujoco.MjData(model)


pos1 = [0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633, 0] # 夹爪向下
pos2 = [0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633, 0] # 夹爪水平
pos3 = [0, -0.382, 3.14159265, -1.75, 0, -1.76, 1.57079633, 0] # 夹爪向下 2
data.ctrl[:] = pos3

dt = model.opt.timestep
duration = 2
steps = int(duration / dt)          # 2 秒对应的步数


# -------------- 模型随机生成 -------------
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



# --------------- 线性流程 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


