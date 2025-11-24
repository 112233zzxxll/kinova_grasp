from xml.etree.ElementInclude import default_loader

import mujoco
import mujoco.viewer
import numpy as np
import time
import glfw
import cv2

# --------------- 基本配置 ---------------
model = mujoco.MjModel.from_xml_path('mixed_model/scene.xml')
data  = mujoco.MjData(model)

bent = np.array([1 ,0.743, 0.59, 1.24, -0.41, 0.882, -0.53, 0])
default = np.array([0, 0, 0, 0, 0, 0, 0 , 0])

dt = model.opt.timestep
duration = 2
steps = int(duration / dt)          # 2 秒对应的步数

# --------------- 相机预设置 ---------------
resolution = (320, 240)
# 创建OpenGL上下文（离屏渲染）
glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
glfw.make_context_current(window)

scene = mujoco.MjvScene(model, maxgeom=10000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

# 设置相机参数
camera_name = "rgb_camera"
camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
camera.fixedcamid = camera_id

# 创建帧缓冲对象
framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
# --------------- 相机预设置 ---------------

# --------------- 线性流程 ---------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    data.ctrl[:] = bent
    while viewer.is_running():
        # --------------- 相机设置模块( ---------------
        viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), mujoco.MjvPerturb(), camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, viewport, context)
        # 转换颜色空间 (OpenCV使用BGR格式)
        bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        cv2.imshow('MuJoCo Camera Output', bgr)
        viewer.sync()
        cv2.waitKey(1)
        # --------------- )相机设置模块 ---------------
        mujoco.mj_step(model, data)
        viewer.sync()
    # --------------- 删除相机资源 ---------------
    cv2.destroyAllWindows()
    glfw.terminate()
    del context
    del scene