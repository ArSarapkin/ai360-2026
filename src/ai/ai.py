import numpy as np
import openai
from openai import OpenAI
from PIL import Image
import io
import base64
import json
import re
import common.pointcloud as pc

from common.pointcloud import BBox3D

api_token = "-"
base_url = "https://api.mulerouter.ai/vendors/openai/v1"

client = OpenAI(
    api_key=api_token,
    base_url=base_url,
)


def ask(text: str, image: Image.Image, model: str = "qwen-vl-max"):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
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


def detect(target: str, img: Image.Image, intrinsics: pc.Intrinsics):
    prompt = build_detect_prompt(target, intrinsics)
    response = ask(prompt, img)
    text = response.strip()
    start = text.find('{')
    end = text.rfind('}')
    data = json.loads(text[start:end + 1])
    return list(data["bbox_3d"])


def angles_to_rotation(angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = angle_x * np.pi, angle_y * np.pi, angle_z * np.pi
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def detect_bbox(target: str, img: Image.Image, intrinsics: pc.Intrinsics):
    detection = detect(target, img, intrinsics)
    position = np.array(detection[0:3])
    size = np.array(detection[3:6])
    roll, pitch, yaw = detection[6], detection[7], detection[8]
    rotation = angles_to_rotation(roll, pitch, yaw)
    return BBox3D(position=position, size=size, rotation=rotation)
