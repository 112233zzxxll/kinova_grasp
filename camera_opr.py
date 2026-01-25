import threading
import mujoco
from queue import Queue

class AsyncRenderer:
    def __init__(self, model, data, cameras=[0, 1], height=64, width=64):
        self.model = model
        self.data = data
        self.cameras = cameras
        self.height = height
        self.width = width
        
        self.render_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        # 每个线程必须有自己的 Renderer 和 MjData
        local_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        local_data = mujoco.MjData(self.model)  # 使用相同的 model

        while True:
            cmd = self.render_queue.get()
            if cmd == "stop":
                break

            # 复制主线程 data 的状态（只读快照）
            local_data.qpos[:] = self.data.qpos
            local_data.qvel[:] = self.data.qvel
            local_data.act[:] = self.data.act
            # 如果你的模型用到了 mocap 或其他状态，也需复制：
            local_data.mocap_pos[:] = self.data.mocap_pos
            local_data.mocap_quat[:] = self.data.mocap_quat

            # 渲染
            images = []
            for cam_id in self.cameras:
                local_renderer.update_scene(local_data, camera=cam_id)
                images.append(local_renderer.render())

            self.result_queue.put(images)

    def render_async(self):
        self.render_queue.put("render")
        return self.result_queue.get()  # 阻塞直到图像就绪

    def close(self):
        self.render_queue.put("stop")
        self.thread.join(timeout=2)  # 等待线程结束