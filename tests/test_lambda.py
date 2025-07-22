import asyncio
import logging
from contextlib import nullcontext
from functools import partial
from typing import Any, ContextManager, Tuple

import cv2
import h5pyd
import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI

from fastapi_h5 import router


@pytest.mark.asyncio
async def test_lambda() -> None:
    app = FastAPI()

    app.include_router(router, prefix="/results")

    def get_data() -> Tuple[dict[str, Any], ContextManager[Any]]:
        def get_42():
            return 42

        def full_image() -> np.array:
            return np.ones((1000, 1000))

        def never_run() -> None:
            raise Exception

        def scale_image(factor):
            img = full_image()
            new_size = (np.array(img.shape) / factor).astype(np.int64)
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            return img

        data = {
            "live": 34,
            "image": np.ones((1000, 1000)),
            "danger": never_run,
            "image_scaled": {str(i): partial(scale_image, i) for i in range(2, 10)},
            "double": get_42,
        }
        return data, nullcontext()  # type: ignore

    app.state.get_data = get_data

    config = uvicorn.Config(app, port=5000, log_level="debug")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    while server.started is False:
        await asyncio.sleep(0.1)

    def work() -> None:
        f = h5pyd.File(
            "/", "r", endpoint="http://localhost:5000/results", timeout=1, retries=0
        )
        logging.info("live %s", f["live"][()])
        logging.info("keys %s", list(f.keys()))
        assert "danger" in f.keys()
        assert f["image_scaled/5"].shape == (200, 200)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, work)

    server.should_exit = True
    await server_task

    await asyncio.sleep(0.5)
