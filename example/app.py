import logging
import time
from contextlib import nullcontext
from functools import partial
from typing import Any, ContextManager, Tuple

import cv2
import numpy as np
from fastapi import FastAPI

from fastapi_h5 import router

app = FastAPI()

app.include_router(router)

logging.basicConfig(level=logging.INFO)


def get_data() -> Tuple[dict[str, Any], ContextManager[None]]:
    dt = np.dtype({"names": ["a", "b"], "formats": [float, int]})

    arr = np.array([(0.5, 1)], dtype=dt)

    def get_42():
        return 42

    def full_image() -> np.array:
        return np.ones((1000, 1000))

    def scale_image(factor):
        img = full_image()
        new_size = (np.array(img.shape) / factor).astype(np.int64)
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        return img

    changing = np.ones((5, int(time.time()) % 10 + 5))
    data = {
        "map": {
            "x": [
                np.float64(-14.96431272),
                np.float64(-14.76401095),
                np.float64(-14.56276384),
                np.float64(-14.361908),
                np.float64(-14.16112703),
            ],
            "y": [
                np.float64(-10.00637736),
                np.float64(-10.00502708),
                np.float64(-10.00403313),
                np.float64(-10.00349819),
                np.float64(-10.00320074),
            ],
            "values": [
                np.float32(0.6831444),
                np.float32(0.0),
                np.float32(0.039953336),
                np.float32(0.14946304),
                np.float32(0.0),
            ],
        },
        "live": 34,
        "calculated": get_42,
        "other": {"third": [1, 2, 3]},  # only _attrs allowed in root
        "other_attrs": {"NX_class": "NXother"},
        "image": np.ones((1000, 1000)),
        "image_scaled": {str(i): partial(scale_image, i) for i in range(2, 10)},
        "image_attrs": {"listattr": [42, 43, 44, 45]},
        "changing_shape": changing,
        "specialtyp": np.ones((10, 10), dtype=">u8"),
        "specialtyp_attrs": {"NXdata": "NXspecial"},
        "hello": "World",
        "_attrs": {"NX_class": "NXentry"},
        "composite": arr,
        "composite_attrs": {"axes": ["I", "q"]},
        "oned_attrs": {"NX_class": "NXdata", "axes": ["motor"], "signal": "data"},
        "oned": {
            "data": np.ones((42)),
            "data_attrs": {"long_name": "photons"},
            "motor": np.linspace(0, 1, 42),
            "motor_attrs": {"long_name": "motor name"},
        },
        "twod_attrs": {
            "NX_class": "NXdata",
            "axes": ["motor", "."],
            "signal": "frame",
            "interpretation": "image",
        },
        "twod": {
            "frame": np.ones((42, 42)),
            "motor": np.linspace(0, 1, 42),
            "motor_attrs": {"long_name": "motor name"},
        },
    }
    return data, nullcontext()


app.state.get_data = get_data
