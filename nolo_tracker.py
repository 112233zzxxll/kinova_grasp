import ctypes
import time
from ctypes import Structure, c_float, c_int, c_uint, c_byte, POINTER, CFUNCTYPE, byref
import numpy as np
import mink


# 定义NOLO SDK中的数据结构（与C/C++结构体对应）
class NVector2(Structure):
    _fields_ = [("x", c_float), ("y", c_float)]

    def __str__(self):
        return f"({self.x}, {self.y})"
    __repr__ = __str__


class NVector3(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float)]

    def __str__(self):
        return f"V:({self.x},{self.y},{self.z})"
    __repr__ = __str__


class NQuaternion(Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("z", c_float), ("w", c_float)]

    def __str__(self):
        return f"Q:({self.w},{self.x},{self.y},{self.z})"
    __repr__ = __str__


class Controller(Structure):
    _fields_ = [
        ("VersionID", c_int),
        ("Position", NVector3),
        ("Rotation", NQuaternion),
        ("Buttons", c_uint),
        ("Touched", c_int),
        ("TouchAxis", NVector2),
        ("Battery", c_int),
        ("State", c_int),
    ]


class HMD(Structure):
    _fields_ = [
        ("HMDVersionID", c_int),
        ("HMDPosition", NVector3),
        ("HMDInitPostion", NVector3),
        ("HMDTwoPointDriftAngle", c_uint),
        ("HMDRotation", NQuaternion),
        ("HMDState", c_int),
    ]


class BaseStation(Structure):
    _fields_ = [("BaseStationVersionID", c_int), ("BaseStationPower", c_int)]


# 由于NoloSensorData和NOLOData包含数组，需要特殊处理
class NOLOData(Structure):
    pass  # 需要先定义，因为Controller中可能引用它


# 设置NOLOData的字段（需要后向引用）
NOLOData._fields_ = [
    ("leftData", Controller),
    ("rightData", Controller),
    ("hmdData", HMD),
    ("bsData", BaseStation),
    ("expandData", c_byte * 64),  # 64字节数组
    # NoloSensorData结构省略，因为我们主要关注手柄数据
    ("leftPackNumber", c_byte),
    ("rightPackNumber", c_byte),
    ("FixedEyePosition", NVector3),
]

# 定义回调函数类型
# 对应C++中的: typedef void(__cdecl *pfnDataCallBack)(const NOLOData &noloData);
DATA_CALLBACK = CFUNCTYPE(None, POINTER(NOLOData))


# 定义回调类型枚举（与C#中的ECallBackTypes一致）
class ECallBackTypes:
    eOnZMQConnected = 0
    eOnZMQDisConnected = 1
    eOnButtonDoubleClicked = 2
    eOnKeyPressEvent = 3
    eOnKeyReleaseEvent = 4
    eOnNewData = 5  # 我们主要关注这个回调类型
    eOnNoloDevVersion = 6
    eCallBackCount = 7


# 加载NOLO SDK DLL

# 根据您的系统环境，可能需要调整DLL路径
nolo_lib = ctypes.WinDLL("./NoloClient/lib64/NoloClientLib.dll")

# 定义函数原型
nolo_lib.OpenNoloZeroMQ.restype = ctypes.c_bool
nolo_lib.CloseNoloZeroMQ.restype = None
nolo_lib.RegisterCallBack.restype = None
nolo_lib.SetHmdCenter.restype = None


button = None
d_so3 = np.array([1,0,0,0], dtype=np.float64)
d_so3 = mink.SO3(d_so3)  # 构造 SO3 群元
d_pos = [0,0,0]
old_pos = None
old_so3 = None
# 定义回调函数
def on_new_data(nolo_data_ptr):
    """处理新数据的回调函数"""
    global button, d_so3, d_pos, old_pos, old_so3
    # 获取手柄动态
    nolo_data = nolo_data_ptr.contents
    pos = np.frombuffer(nolo_data.leftData.Position, dtype=np.float32) # 新平移
    x = pos[2]
    y = -pos[0]
    z = pos[1]
    pos = np.array([x, y, z], dtype=np.float32)
    # print(f"\r{pos} ", end="", flush=True)

    rot = np.frombuffer(nolo_data.leftData.Rotation, dtype=np.float32) # 新旋转四元数
    wr = rot[3]
    xr = -rot[2]
    yr = rot[0]
    zr = -rot[1]
    rot = np.array([wr,xr,yr,zr], dtype=np.float64)
    # print(f"\r{rot}", end="", flush=True)
    # map_so3 = mink.SO3.from_matrix(np.array([[0, 0, 1],
    #                                          [1, 0, 0],
    #                                          [0, 1, 0]], dtype=np.float64)).inverse()
    # print(f"\r{rot} ", end="", flush=True)
    curr_so3 = mink.SO3(rot) # 新so3
    # curr_so3 = curr_so3.multiply(map_so3)

    button = nolo_data.leftData.Buttons # 按钮

    if old_pos is None:
        d_pos = pos - pos
        d_so3 = curr_so3.multiply(curr_so3.inverse())
        old_pos = d_pos
        old_so3 = d_so3
    else:
        d_pos = pos - old_pos
        d_so3 = curr_so3.multiply(old_so3.inverse())
    old_pos = pos.copy()
    old_so3 = curr_so3.copy()




# 保持对回调函数的引用，防止被垃圾回收
data_callback_func = DATA_CALLBACK(on_new_data)

# 定义连接和断开回调（可选）
def on_zmq_connected():
    print("设备已连接")
    # 可以在这里设置头显中心点
    hmd_center = NVector3(0.0, 0.09, 0.07)
    nolo_lib.SetHmdCenter(byref(hmd_center))

def on_zmq_disconnected():
    print("设备已断开")

connected_callback = CFUNCTYPE(None)(on_zmq_connected)
disconnected_callback = CFUNCTYPE(None)(on_zmq_disconnected)

# 保持对这些回调的引用
callbacks = [connected_callback, disconnected_callback, data_callback_func]

# 初始化并注册回调
def init_nolo():
    # 打开连接
    if nolo_lib.OpenNoloZeroMQ():
        print("成功打开NOLO连接")

        # 注册回调
        nolo_lib.RegisterCallBack(
            ECallBackTypes.eOnZMQConnected,
            ctypes.cast(connected_callback, ctypes.c_void_p),
        )
        nolo_lib.RegisterCallBack(
            ECallBackTypes.eOnZMQDisConnected,
            ctypes.cast(disconnected_callback, ctypes.c_void_p),
        )
        nolo_lib.RegisterCallBack(
            ECallBackTypes.eOnNewData,
            ctypes.cast(data_callback_func, ctypes.c_void_p),
        )
        return True
    else:
        print("打开NOLO连接失败")
        return False

    # 主程序
def func():
    print("是否运行：NoloDeviceSDK-master/NoloServer/NoloServer.exe")
    print("开始连接NOLO设备...")
    init_nolo()

def get_delta():
    return d_pos, d_so3, button


