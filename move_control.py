import mujoco.viewer
import numpy as np
import mink
import time
import glfw
import cv2

model1 = mujoco.MjModel.from_xml_path("mixed_model/gen3.xml")
data1 = mujoco.MjData(model1) # 求解空间

model2 = mujoco.MjModel.from_xml_path("mixed_model/scene.xml")
data2 = mujoco.MjData(model2) # 显示

ee_pos = np.array([0.1, -0.58, 0.4])
ee_rot = np.array([[0.5,0.866,0], [0.866,-0.5,0], [0,0,-1]])
rot = mink.SO3.from_matrix(ee_rot)
target = mink.SE3.from_rotation_and_translation(rot, ee_pos) # 对准，这个 target 后续切换成物体位位姿估计

ee_pos1 = np.array([0.1, -0.58, 0.26])
target1 = mink.SE3.from_rotation_and_translation(rot, ee_pos1) # 靠近




configuration1 = mink.Configuration(model1)
configuration2 = mink.Configuration(model2)

tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="pinch_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model1, cost=1e-3),
    ] # 运动优化目标

collision_pairs = [(["red_box_geom"], ["floor"])]

limits = [
    mink.ConfigurationLimit(model=model1),
    mink.CollisionAvoidanceLimit(model=model1, geom_pairs=collision_pairs),
] # 碰撞边界条件以及关节角限制

max_velocities = {
        "joint_1": np.pi/6,
        "joint_2": np.pi/6,
        "joint_3": np.pi/3,
        "joint_4": np.pi/3,
        "joint_5": np.pi/3,
        "joint_6": np.pi/3,
        "joint_7": np.pi/3,
                }  # 关节限速

velocity_limit = mink.VelocityLimit(model1, max_velocities)
limits.append(velocity_limit)

model = configuration1.model
data = configuration1.data # 提取数据，准备求解
solver = "daqp"

# # --------------- 相机预设置 ---------------
# resolution = (320, 240)
# # 创建OpenGL上下文（离屏渲染）
# glfw.init()
# glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
# window = glfw.create_window(resolution[0], resolution[1], "Offscreen", None, None)
# glfw.make_context_current(window)
#
# scene = mujoco.MjvScene(model2, maxgeom=10000)
# context = mujoco.MjrContext(model2, mujoco.mjtFontScale.mjFONTSCALE_150.value)
#
# # 设置相机参数
# camera_name = "rgb_camera"
# camera_id = mujoco.mj_name2id(model2, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
# camera = mujoco.MjvCamera()
# camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
# camera.fixedcamid = camera_id
#
# # 创建帧缓冲对象
# framebuffer = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
# mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context)
# # --------------- 相机预设置 ---------------

with mujoco.viewer.launch_passive(model2, data2) as viewer:
    data2.ctrl[:7] = [0, 0, 0, 0, 0, 0, 0]
    configuration1.update_from_keyframe("home")
    posture_task.set_target(configuration1.q)

    i = 1 # 对准阶段

    while viewer.is_running():

        # # --------------- 相机设置模块( ---------------
        # viewport = mujoco.MjrRect(0, 0, resolution[0], resolution[1])
        # mujoco.mjv_updateScene(model2, data2, mujoco.MjvOption(), mujoco.MjvPerturb(), camera, mujoco.mjtCatBit.mjCAT_ALL,
        #                        scene)
        # mujoco.mjr_render(viewport, scene, context)
        # rgb = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        # mujoco.mjr_readPixels(rgb, None, viewport, context)
        # # 转换颜色空间 (OpenCV使用BGR格式)
        # bgr = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
        # cv2.imshow('MuJoCo Camera Output', bgr)
        # viewer.sync()
        # cv2.waitKey(1)
        # # --------------- )相机设置模块 ---------------

        if i == 1: # 对准阶段
            target = target # 这里用于接入位姿估计实时更新 target（物体位置）
            end_effector_task.set_target(target) # 设定 target 求解
            vel = mink.solve_ik(configuration1, tasks, 0.002, solver, limits=limits) # 求解关节速度
            configuration1.integrate_inplace(vel, 0.002) # 关节角位置
            data2.ctrl[:7] = configuration1.q # 更新控制
            mujoco.mj_step(model2, data2)
            viewer.sync()

            # 下面进行误差计算
            T_ee = configuration1.get_transform_frame_to_world("pinch_site", "site") # 实时末端位姿
            T_diff = T_ee.inverse() @ target # 实际姿态和目标姿态的重合程度->SE3
            T_diff = mink.lie.se3.SE3.as_matrix(T_diff) - np.identity(4) # SE3 -> ndarray
            norm_fro = np.linalg.norm(T_diff, 'fro') # 矩阵范数做误差

            # 误差判断
            if norm_fro <= 1e-5:
                i = 2 # 进入接近阶段
                old_target = target # 记录 target，防止阶段二 target 位置变化，进入阶段一重新对准
                print("阶段2")
                # 靠近，就是夹爪沿着 z 平移-0.14，重新生成 target1 位姿
                ee_pos[2] = ee_pos[2] - 0.13
                target1 = mink.SE3.from_rotation_and_translation(rot, ee_pos)
                norm_fro = 1 # 初始化误差，防止回到阶段1后跳过
            else:
                i = 1


        elif i == 2:
            target = target  # 依然实时更新 target（物体位置）
            T_diff1 = old_target.inverse() @ target  # 物体现位置和原位置的重合程度->SE3
            T_diff1 = mink.lie.se3.SE3.as_matrix(T_diff1) - np.identity(4) # SE3 -> ndarray
            norm_fro1 = np.linalg.norm(T_diff1, 'fro')  # 矩阵范数做误差
            if norm_fro1 <= 3e-6:
                i = 2 # 保持接近阶段
            else:
                i = 1 # 切换回对准阶段
                norm_fro1 = 1 # 初始化误差
                print("回到1阶段")

            T_ee0 = configuration1.get_transform_frame_to_world("pinch_site", "site")  # 实时末端位姿
            T_diff0 = T_ee0.inverse() @ target1  # 实际姿态和目标姿态的重合程度 -> SE3
            T_diff0 = mink.lie.se3.SE3.as_matrix(T_diff0) - np.identity(4)  # SE3 -> ndarray
            norm_fro0 = np.linalg.norm(T_diff0, 'fro')  # 矩阵范数做误差
            if norm_fro0 <= 1e-5:
                i = 3
                print("阶段3")
                norm_fro0 = 1
            else:
                i = 2

            end_effector_task.set_target(target1)  # 设定 target1 求解
            vel = mink.solve_ik(configuration1, tasks, 0.002, solver, limits=limits)  # 求解关节速度
            configuration1.integrate_inplace(vel, 0.002)  # 关节角位置
            data2.ctrl[:7] = configuration1.q  # 更新控制
            mujoco.mj_step(model2, data2)
            viewer.sync()

        elif i == 3:
            mujoco.mj_step(model2, data2)
            viewer.sync()
             # 手动闭合夹爪