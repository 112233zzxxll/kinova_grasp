import nolo_tracker
from mink_use import get_target
from mink import SO3
import mujoco
import mujoco.viewer
import numpy as np
import time


model = mujoco.MjModel.from_xml_path('mixed_model/scene.xml')
data  = mujoco.MjData(model)

pos = [0, -0.382, 3.14159265, -1.75, 0, -1.43, 1.57079633, 0] # 夹爪斜向下
data.ctrl[:] = pos

nolo_tracker.func()
old_ee_pos = np.array([0.5,0,0.1])
old_ee_rot = np.array([1,0,0,0], dtype=np.float64)
old_ee_rot = SO3(old_ee_rot)  # 构造 SO3 群元


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        d_pos, d_so3, button = nolo_tracker.get_delta()
        if button>0:
            old_ee_pos, old_ee_rot, target = get_target(old_ee_pos, old_ee_rot, d_pos, d_so3)
        # 已知新位置 target_pos: ndarray(3,)
            model.body("target").pos = old_ee_pos  # 立即生效
            quat = old_ee_rot.parameters()
            # print(f"\r{quat} ", end="", flush=True)
            model.body("target").quat = quat
            # print(f"\r{old_ee_pos} ", end="", flush=True)