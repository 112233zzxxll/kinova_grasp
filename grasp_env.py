import numpy as np
import mujoco
import mink

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