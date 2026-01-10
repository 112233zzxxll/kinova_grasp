import mujoco
import mujoco.viewer
import numpy as np
import time

# --------------- 基本配置 ---------------
model = mujoco.MjModel.from_xml_path('mixed_model/scene.xml')
data  = mujoco.MjData(model)





# --------------- 线性流程 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


