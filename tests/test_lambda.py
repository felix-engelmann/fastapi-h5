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
import random

from fastapi_h5 import router
from fastapi_h5.h5types import H5SimpleShape, H5CalculatedDataset
from fastapi_h5.utils import _canonical_to_h5


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

@pytest.mark.asyncio
async def test_caching() -> None:
    app = FastAPI()

    app.include_router(router, prefix="/results")

    def get_data() -> Tuple[dict[str, Any], ContextManager[Any]]:

        def rand():
            return np.ones((100,100))* random.random()

        data = {
            "random": rand
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
        assert f["random"].shape == (100, 100)
        full = f["random"][:]
        # fetching the full dataset at once is a single call to the function.
        assert np.array_equal(full, np.ones((100,100))*full[0,0])

        first = f["random"][:50]
        second = f["random"][50:]
        # every value call, event to a slice calls the function again.
        assert not np.array_equal(first, second)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, work)

    server.should_exit = True
    await server_task

    await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_fancy() -> None:
    app = FastAPI()

    app.include_router(router, prefix="/results")

    def get_data() -> Tuple[dict[str, Any], ContextManager[Any]]:

        def get_ds():
            arr = np.ones((10, 10))

            h5shape = H5SimpleShape(dims=list(arr.shape))

            canonical = arr.dtype.descr[0][1]
            h5type = _canonical_to_h5(canonical)

            return H5CalculatedDataset(shape=h5shape, type=h5type, get_value=lambda: arr)

        def bad():
            raise Exception("should never be called")

        def get_meta():
            h5shape = H5SimpleShape(dims=[12,12])
            h5type = _canonical_to_h5('<f8')
            return H5CalculatedDataset(shape=h5shape, type=h5type, get_value=bad)

        data = {
            "fullds": get_ds,
            "onlymeta": get_meta
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
        assert f["fullds"].shape == (10, 10)
        assert np.array_equal(f["fullds"], np.ones((10,10)))

        assert f["onlymeta"].shape == (12, 12)
        assert f["onlymeta"].dtype == np.float64

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, work)

    server.should_exit = True
    await server_task

    await asyncio.sleep(0.5)
