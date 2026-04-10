import os
import threading

import numpy as np
import requests

SPIRE_PORT = os.getenv("SPIRE_PORT", "8000")
SPIRE_HOST = os.getenv("SPIRE_HOST", "localhost")

_thread_local = threading.local()


def _get_session() -> requests.Session:
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        _thread_local.session = s
    return s


def health_check(port: str = SPIRE_PORT, host: str = SPIRE_HOST, timeout_s: float = 2.0) -> bool:
    url = f"http://{host}:{port}/healthz"
    r = _get_session().get(url, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("status") == "ok"

def generate_spire_image(
    W: int = 256,
    structure_type: int = 1,
    uc_scale_ab: float = 1.0,
    uc_scale_c: float = 1.0,
    channel_vol_prop: float = 0.5,
    slice_height: float = 1.0,
    slice_width: float = 1.0,
    slice_thickness: float = 1.0,
    slice_position: float = 0.0,
    h: int = 0,
    k: int = 0,
    l: int = 1,
    membrane_distance: float = 0.0,
    membrane_thickness: float = 0.02,
    image_depth: int = 76,
    port = SPIRE_PORT,
    timeout_s: float = 120.0,
    keepalive: bool = False,
):
    url = f"http://{SPIRE_HOST}:{port}/compute/generate_spire_image"
    params = {
        "W": W,
        "structure_type": structure_type,
        "uc_scale_ab": uc_scale_ab,
        "uc_scale_c": uc_scale_c,
        "channel_vol_prop": channel_vol_prop,
        "slice_height": slice_height,
        "slice_width": slice_width,
        "slice_thickness": slice_thickness,
        "slice_position": slice_position,
        "h": h,
        "k": k,
        "l": l,
        "membrane_distance": membrane_distance,
        "membrane_thickness": membrane_thickness,
        "image_depth": image_depth,
    }
    headers = None if keepalive else {"Connection": "close"}
    response = _get_session().post(url, params=params, timeout=timeout_s, headers=headers)
    response.raise_for_status()

    # Extract image dimensions from headers
    width = int(response.headers["X-Image-Width"])
    height = int(response.headers["X-Image-Height"])
    dtype = response.headers["X-DType"]

    image = np.frombuffer(response.content, dtype=dtype).reshape((height, width))
    return image