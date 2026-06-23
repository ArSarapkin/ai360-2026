import os
import time
import cupy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from scannet.scannet_config import ScannetScene

data_path = '../../data/scannet'

def scene_pointcloud(scene_path, frame_id_limit=0, num_workers=8):
    scene_path = f'{data_path}/posed_images/{scene_path}'
    scannet_scene = ScannetScene(scene_path)

    frame_ids = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(scene_path)
        if f.endswith('.jpg')
    )
    if frame_id_limit > 0:
        frame_ids = [f for f in frame_ids if int(f) <= frame_id_limit]
    frame_ids = frame_ids[::5]

    def process_frame(frame_id):
        start_time = time.time()
        print(f"Starting processing frame {frame_id}")
        scene = scannet_scene.build_scene(frame_id)
        frame = scannet_scene.load_frame(frame_id)
        result = scene.process_frame(frame)
        print(f"Finished processing frame {frame_id} in {time.time() - start_time}")
        return frame_id, result

    chunks = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_frame, fid): fid for fid in frame_ids}
        for future in as_completed(futures):
            try:
                fid, result = future.result()
                chunks[fid] = result
            except Exception as e:
                print(f"Frame {futures[future]} failed: {e}")

    ordered = [chunks[fid] for fid in sorted(chunks)]
    return np.concatenate(ordered, axis=0)
