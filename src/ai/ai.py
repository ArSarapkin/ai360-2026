import time

import numpy as np
import cupy as cp
import openai
from openai import OpenAI
import json
import re
import common.pointcloud as pc

from common.pointcloud import BBox3D

api_token = "sk-mr-5d314a41ac90ce0d25119dd73969c4e5dea255eec4993fac590b88902c596d26"
base_url = "https://api.mulerouter.ai/vendors/openai/v1"
local_base_url = "http://127.0.0.1:1234/v1"

client = OpenAI(
    api_key=api_token,
    base_url=local_base_url,
)


def ask(text: str, image_base64: str, model: str = "qwen-vl-max"):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {"type": "text", "text": text},
            ]
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content


output_format = "`{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]}`"


def build_detect_prompt(target: str, intrinsics: pc.Intrinsics) -> str:
    return (
        f"Find the {target} in this image and output ONE 3D bounding box as JSON: "
        f"{output_format}\n"
        f"Intrinsics: fx={intrinsics.fx}, fy={intrinsics.fy}, "
        f"cx={intrinsics.cx}, cy={intrinsics.cy}.\n"
    )


def _parse_detections(text: str) -> list:
    first_bracket = text.find('[')
    first_brace = text.find('{')
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        start, end = first_bracket, text.rfind(']')
        raw = json.loads(text[start:end + 1])
        return [item["bbox_3d"] for item in raw]
    else:
        start, end = first_brace, text.rfind('}')
        raw = json.loads(text[start:end + 1])
        return [raw["bbox_3d"]]


def detect(target: str, img_base64: str, intrinsics: pc.Intrinsics) -> list:
    prompt = build_detect_prompt(target, intrinsics)
    response = ask(prompt, img_base64)
    text = response.strip()
    try:
        return _parse_detections(text)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"[detect:{target}] JSON error: {e}\nRaw response: {text}")
        return []


def angles_to_rotation(angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = angle_x * np.pi, angle_y * np.pi, angle_z * np.pi
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def detect_bbox(target: str, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
    t_start = time.time()
    detections = detect(target, img_base64, intrinsics)
    bboxes = []
    for detection in detections:
        position = cp.array(detection[0:3])
        size = cp.array(detection[3:6])
        roll, pitch, yaw = detection[6], detection[7], detection[8]
        rotation = cp.asarray(angles_to_rotation(roll, pitch, yaw))
        bboxes.append(BBox3D(position=position, size=size, rotation=rotation))
    t_stop = time.time()
    print(f"[{target}] detected {len(bboxes)}, time: {t_stop - t_start:.2f}s")
    return bboxes
