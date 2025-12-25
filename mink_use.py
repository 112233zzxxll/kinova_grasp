import mujoco.viewer
import numpy as np
import mink
from mink import SO3
import time
import glfw
import cv2

def init_model():
    model = mujoco.MjModel.from_xml_path("mixed_model/gen3.xml")
    data = mujoco.MjData(model) # 求解空间

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="pinch_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]  # 运动优化目标

    collision_pairs = [(["red_box_geom"], ["floor"])] # 碰撞对

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]  # 碰撞边界条件以及关节角限制

    max_velocities = {
        "joint_1": np.pi / 6,
        "joint_2": np.pi / 6,
        "joint_3": np.pi / 3,
        "joint_4": np.pi / 3,
        "joint_5": np.pi / 3,
        "joint_6": np.pi / 3,
        "joint_7": np.pi / 3,
    }  # 关节限速

    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    solver = "daqp"

    configuration.update_from_keyframe("home")
    posture_task.set_target(configuration.q)

    return configuration, tasks, end_effector_task, solver, limits, model, data


def get_target(old_ee_pos, old_ee_rot, d_pos, d_so3): #通过 nolo 计算机械臂末端相对位移

    new_ee_pos = old_ee_pos + d_pos

    new_ee_rot = d_so3.multiply(old_ee_rot)
    # new_ee_rot = d_so3.log() + old_ee_rot.log()
    # new_ee_rot = SO3.exp(new_ee_rot)

    target = mink.SE3.from_rotation_and_translation(new_ee_rot, new_ee_pos)  # 对准，这个 target 后续切换成物体位位姿估计

    return new_ee_pos, new_ee_rot, target


def solve(configuration, tasks, end_effector_task, solver, limits, model, data, target):
    end_effector_task.set_target(target)  # 设定 target 求解
    vel = mink.solve_ik(configuration, tasks, 0.002, solver, limits=limits)  # 求解关节速度
    configuration.integrate_inplace(vel, 0.002)  # 关节角位置
    return configuration.q  # 更新控制







