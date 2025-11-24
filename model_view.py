import mujoco
import mujoco.viewer
import numpy as np
import random
import time

# --------------- 基本配置 ---------------
model = mujoco.MjModel.from_xml_path('mixed_model/gen3.xml')
data  = mujoco.MjData(model)


pos1 = [0, -0.34906585, 3.14159265, -2.54818071, 0, -0.87266463, 1.57079633]
pos2 = [0, 0.26179939, 3.14159265, -2.26892803, 0, 0.95993109, 1.57079633]
data.ctrl[:] = pos2

dt = model.opt.timestep
duration = 2
steps = int(duration / dt)          # 2 秒对应的步数
step = 0 # step = 300 可让机械臂到 pos2


# --------------- 线性流程 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


