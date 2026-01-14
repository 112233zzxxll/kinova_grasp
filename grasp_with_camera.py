import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time

# ----------------------------
# 1. 加载模型（确保 XML <size> 已正确设置）
# ----------------------------
model = mujoco.MjModel.from_xml_path("mixed_model/scene.xml")
data = mujoco.MjData(model)

print(f"✅ njmax={model.njmax}, nconmax={model.nconmax}")

# ----------------------------
# 2. 获取相机 ID
# ----------------------------
try:
    cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
    cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
except:
    print("⚠️ 相机未找到，使用索引 0 和 1")
    cam1_id, cam2_id = 0, 1

# ----------------------------
# 3. 创建离屏渲染器
# ----------------------------
renderer = mujoco.Renderer(model, height=224, width=224)

# ----------------------------
# 4. 启动交互式 viewer（单线程主循环）
# ----------------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # ✅ 在主线程中推进仿真（关键！）
        mujoco.mj_step(model, data)

        # ✅ 渲染两个离屏视角
        renderer.update_scene(data, camera=cam1_id)
        img1 = renderer.render()

        renderer.update_scene(data, camera=cam2_id)
        img2 = renderer.render()

        # 显示离屏图像
        cv2.imshow("Top View", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
        cv2.imshow("Side View", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)

        # 同步交互式 viewer
        viewer.sync()

        # 控制帧率（可选）
        time_until_next = model.opt.timestep - (time.time() - step_start)
        if time_until_next > 0:
            time.sleep(time_until_next)

cv2.destroyAllWindows()