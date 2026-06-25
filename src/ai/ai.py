import time
import threading

import httpx
import numpy as np
from openai import OpenAI

api_token = "sk-mr-5d314a41ac90ce0d25119dd73969c4e5dea255eec4993fac590b88902c596d26"
local_base_url = "http://45.157.161.245:8080/v1"


class _LoggingTransport(httpx.HTTPTransport):
    def handle_request(self, request):
        t = time.time()
        print(f"[conn] [{threading.current_thread().name}] connecting")
        response = super().handle_request(request)
        print(f"[conn] [{threading.current_thread().name}] connected: {time.time() - t:.2f}s")
        return response


client = OpenAI(
    api_key=api_token,
    base_url=local_base_url,
    http_client=httpx.Client(transport=_LoggingTransport()),
)


def ask(text: str, image_base64: str, model: str = "qwen-vl-max", temperature: float = 0.0):
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
    print(f"[ask] sending request")
    t = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024 + 512,
        temperature=temperature,
    )
    print(f"[ask] {time.time() - t:.2f}s")
    return response.choices[0].message.content


def angles_to_rotation(angle_x, angle_y, angle_z):
    angle_x, angle_y, angle_z = angle_x * np.pi, angle_y * np.pi, angle_z * np.pi
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    cy, sy = np.cos(angle_y), np.sin(angle_y)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
