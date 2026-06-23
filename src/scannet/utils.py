import time
import cupy as np

from scannet.scannet_config import ScannetScene


def scene_pointcloud(scene_path, frame_id_limit = 0):
    scannet_scene = ScannetScene(scene_path)

    def frame_pointcloud(frame_id):
        start_time = time.time()
        print(f"Starting processing frame {frame_id}")
        scene = scannet_scene.build_scene(frame_id)
        frame = scannet_scene.load_frame(frame_id)
        finish_time = time.time()
        print(f"Finished processing frame in {finish_time - start_time} seconds")
        return scene.process_frame(frame)

    chunks = []
    frame_id = 0
    while True:
        if frame_id > frame_id_limit > 0:
            break
        try:
            frame = str(frame_id).zfill(5)
            chunks.append(frame_pointcloud(frame))
            frame_id += 4
        except:
            break
    return np.concatenate(chunks, axis=0)
