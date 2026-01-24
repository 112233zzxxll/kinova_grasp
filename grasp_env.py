import numpy as np
import mujoco
import mink
import cv2

model = None
data = None
def init_env(path):
    global model, data
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data

def reset_target():
    workspace = {
        'x': [0.3, 0.5],  # -0.6 离底座最近
        'y': [-0.1, 0.5],  # 1.2 为中间
        'z': [0.01, 0.01],  # 固定生成高度
        'theta': [-1.57, 1.57]
    }
    """在工作空间内生成随机位置"""
    x = np.random.uniform(*workspace['x'])
    y = np.random.uniform(*workspace['y'])
    z = np.random.uniform(*workspace['z'])
    theta = np.random.uniform(*workspace['theta'])
    random_pos = np.array([x, y, z])
    random_quat = np.array([np.cos(theta), 0, 0, np.sin(theta)])

    """获取物体id"""
    # object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "six") #修改默认位置
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "six_joint")
    object_qpos_adr = model.jnt_qposadr[object_joint_id]

    """赋予位置"""
    # model.body_pos[object_body_id] = random_pos # 默认位置修改
    data.qpos[object_qpos_adr:object_qpos_adr + 3] = random_pos
    # model.body_quat[object_body_id] = random_quat # 默认位置修改
    data.qpos[object_qpos_adr + 3:object_qpos_adr + 7] = random_quat

    workspace = {
        'x1': [0.4, 0.5],  # -0.6 离底座最近
        'y1': [-0.5, -0.6],  # 1.2 为中间
        'z1': [0.1, 0.4],  # 固定生成高度
    }
    """在工作空间内生成随机位置"""
    x1 = np.random.uniform(*workspace['x1'])
    y1 = np.random.uniform(*workspace['y1'])
    z1 = np.random.uniform(*workspace['z1'])
    random_pos = np.array([x1, y1, z1])

    """获取物体id"""
    object_body_id1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "assemble") #修改默认位置

    """赋予位置"""
    model.body_pos[object_body_id1] = random_pos # 默认位置修改


def reset_ee():
    ee_pos = np.array([0.2, 0, 0.5])
    ee_rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    ee_rot = mink.SO3.from_matrix(ee_rot)
    target = mink.SE3.from_rotation_and_translation(ee_rot, ee_pos)
    return ee_pos, ee_rot, target

def reset_arm():
    return [0.358, -0.279, -0.323, 1.91, -0.0877, 1.5, -1.52, 0]

def get_reward(old_vel):
    # 获取当前位置、速度，返回奖励、速度
    epsilon_1 = 1
    epsilon_2 = 0.6
    epsilon_3 = 0.5
    pos_dist = []
    vel_dist = []
    site_names = ["ee", "tg1", "tg2", "tg3", "tg4", "tg5", "tg6", "tg7"] # 八个量 * 3
    for name in site_names:
        site_vel = np.zeros(6, dtype=np.float64)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[site_id].copy()
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, site_vel, False)
        vel = site_vel[:3]
        vel_dist.append(vel)
        pos_dist.append(pos)
    vel_dist = np.array(vel_dist)
    pos_dist = np.array(pos_dist)
    old_vel = np.array(old_vel)

    # 奖励 1 ：抓取距离,取最短距离（这里是负距离平方，取最大值作为奖励）
    distance_list = []
    for i in range(1, 7):
        distance = -np.sum((pos_dist[0] - pos_dist[i]) ** 2)
        distance_list.append(distance)
    reward_1 = max(distance_list)

    # 奖励 2 ：夹爪末端加速度以及框架加速度，取七个的最大加速度（这里是加速度的平方，取最大加速度做比较，高于某阈值就惩罚）
    acc_list = []
    reward_2 = 0
    for i in range(0, 7):
        acc = np.sum(((old_vel[i] - vel_dist[i])/2) ** 2)
        acc_list.append(acc)
    max_acc = max(acc_list)
    if max_acc >= 4:
        reward_2 = -(max_acc - 4)
    
    # 奖励 3 ：装配（这里是与装配目标的负距离平方，取最大值作为奖励）
    distance_list = []
    for i in range(1, 7):
        distance = -np.sum((pos_dist[7] - pos_dist[i]) ** 2)
        distance_list.append(distance)
    reward_3 = max(distance_list)

    step_reward = epsilon_1*reward_1 + epsilon_2*reward_2 + epsilon_3*reward_3
    old_vel = vel_dist

    return step_reward, old_vel

# def get_obs():
#     view = True
#     cam1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rgb_camera")
#     cam2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fixed_camera")
#     renderer = mujoco.Renderer(model, height=112, width=112)
#     renderer.update_scene(data, camera=cam1_id)
#     img1 = renderer.render()
#     renderer.update_scene(data, camera=cam2_id)
#     img2 = renderer.render()
#     # 显示离屏图像
#     if view == True:
#         cv2.imshow("Top View", cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
#         cv2.imshow("Side View", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
#         cv2.waitKey(1)
#     return img1, img2