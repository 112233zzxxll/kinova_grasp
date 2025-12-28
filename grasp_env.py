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
        'x': [-0.6, 0],  # -0.6 离底座最近
        'y': [1, 1.4],  # 1.2 为中间
        'z': [1.4, 1.4]  # 固定生成高度
    }
    """在工作空间内生成随机位置"""
    x = np.random.uniform(*workspace['x'])
    y = np.random.uniform(*workspace['y'])
    z = np.random.uniform(*workspace['z'])
    random_pos = np.array([x, y, z])
    # print(random_pos)
    """获取物体id"""
    object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "six")
    object_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "six_joint")
    object_qpos_adr = model.jnt_qposadr[object_joint_id]

    """赋予位置"""
    model.body_pos[object_body_id] = random_pos
    data.qpos[object_qpos_adr:object_qpos_adr + 3] = random_pos
    model.body_quat[object_body_id] = np.array([0,1,0,0])
    print(random_pos)

def reset_ee():
    ee_pos = np.array([0.2, 0, 0.5])
    ee_rot = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    ee_rot = mink.SO3.from_matrix(ee_rot)
    target = mink.SE3.from_rotation_and_translation(ee_rot, ee_pos)
    return ee_pos, ee_rot, target