import copy
import gzip
import logging
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from starlette.requests import Request
from starlette.responses import Response

from fastapi_h5.h5types import (
    H5UUID,
    H5Attribute,
    H5Dataset,
    H5Group,
    H5Link,
    H5Root,
    H5ValuedAttribute,
    H5CalculatedDataset,
)
from fastapi_h5.utils import (
    _dataset_from_obj,
    _get_group_link,
    _get_group_links,
    _get_obj_attrs,
    _path_to_uuid,
    _uuid_to_obj,
)

router = APIRouter()

logger = logging.getLogger()


@router.get("/datasets/{uuid}/value", response_model=None)
def values(
    req: Request, uuid: H5UUID, select: str | None = None
) -> dict[str, Any] | Response:
    data, lock = req.app.state.get_data()
    with lock:
        obj, _ = _uuid_to_obj(data, uuid)
        logger.debug("return value for obj %s", obj)
        logger.debug("selection %s", select)

        if callable(obj):
            obj = obj()
            if isinstance(obj, H5CalculatedDataset):
                obj = obj.get_value()

        if type(obj) in [int, float, str]:
            return {"value": obj}

        elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], str):
            return {"value": copy.copy(obj)}
        else:
            ret = np.array(obj)
            # np.array copes the content of obj, so we can release the lock

    slices = []
    if select is not None:
        dims = select[1:-1].split(",")
        slices = [slice(*map(int, dim.split(":"))) for dim in dims]

    logger.debug("slices %s", slices)

    if len(slices) == 1:
        ret = ret[slices[0]]
    elif len(slices) > 1:
        # TODO: from python 3.11, this can be written as
        # ret = ret[*slices]
        ret = ret[tuple(slices)]

    ret_bytes = ret.tobytes()
    if "gzip" in req.headers.get("Accept-Encoding", ""):
        if len(ret_bytes) > 1000:
            compressed_data = gzip.compress(ret_bytes, compresslevel=7)

            return Response(
                content=compressed_data,
                media_type="application/octet-stream",
                headers={"Content-Encoding": "gzip"},
            )
    return Response(content=ret_bytes, media_type="application/octet-stream")


@router.get("/datasets/{uuid}")
def datasets(req: Request, uuid: H5UUID) -> H5Dataset:
    data, lock = req.app.state.get_data()
    with lock:
        obj, path = _uuid_to_obj(data, uuid)
        logger.debug("return dataset for obj %s", obj)
        ret = _dataset_from_obj(data, path, obj, uuid)
        logger.debug("return dset %s", ret)
        if ret is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return ret


@router.get("/groups/{uuid}/links/{name}")
def link(req: Request, uuid: H5UUID, name: str) -> dict[Literal["link"], H5Link]:
    data, lock = req.app.state.get_data()
    with lock:
        obj, path = _uuid_to_obj(data, uuid)
        path.append(name)
        if name in obj:
            obj = obj[name]
            key: Literal["link"] = "link"
            ret = {key: _get_group_link(obj, path)}
            logger.debug("link name is %s", ret)
            return ret
    raise HTTPException(status_code=404, detail="Link not found")


@router.get("/groups/{uuid}/links")
def links(req: Request, uuid: H5UUID) -> dict[Literal["links"], list[H5Link]]:
    data, lock = req.app.state.get_data()
    with lock:
        obj, path = _uuid_to_obj(data, uuid)
        key: Literal["links"] = "links"
        ret = {key: _get_group_links(obj, path)}
        logger.debug("group links %s", ret)
        return ret


@router.get("/{typ}/{uuid}/attributes/{name}")
def attribute(
    req: Request, typ: Literal["groups", "datasets"], uuid: H5UUID, name: str
) -> H5ValuedAttribute:
    logger.debug("get attr with name %s %s %s", typ, uuid, name)
    data, lock = req.app.state.get_data()
    with lock:
        obj, path = _uuid_to_obj(data, uuid)
        logger.debug(
            "start listing attributes typ %s id %s, obj %s, path: %s",
            typ,
            uuid,
            obj,
            path,
        )
        allattrs = _get_obj_attrs(data, path, include_values=True)
        for attr in allattrs:
            if attr.name == name:
                logger.debug("return attribute %s", attr)
                if isinstance(attr, H5ValuedAttribute):
                    return attr
    raise HTTPException(status_code=404, detail="Attribute not found")


@router.get("/{typ}/{uuid}/attributes")
def attributes(
    req: Request, typ: Literal["groups", "datasets"], uuid: H5UUID
) -> dict[Literal["attributes"], list[H5Attribute]]:
    data, lock = req.app.state.get_data()
    with lock:
        obj, path = _uuid_to_obj(data, uuid)
        logger.debug(
            "start listing attributes typ %s id %s, obj %s, path: %s",
            typ,
            uuid,
            obj,
            path,
        )
        # by calling _get_obj_attrs with False, it should never return H5ValuedAttribute
        attrs: list[H5Attribute] = _get_obj_attrs(data, path, include_values=False)  # type: ignore[assignment]
        return {"attributes": attrs}


@router.get("/groups/{uuid}")
def group(req: Request, uuid: H5UUID) -> H5Group:
    data, lock = req.app.state.get_data()
    with lock:
        obj, path = _uuid_to_obj(data, uuid)
        logger.debug("start listing group id %s, obj %s, path: %s", uuid, obj, path)

        if isinstance(obj, dict):
            linkCount = len(
                list(
                    filter(
                        lambda x: isinstance(x, str) and not x.endswith("_attrs"),
                        obj.keys(),
                    )
                )
            )
            group = H5Group(
                root=_path_to_uuid([], collection_type="g-"),
                id=uuid,
                linkCount=linkCount,
                attributeCount=len(_get_obj_attrs(data, path)),
            )
            logger.debug("group is %s", group)
            return group
    raise HTTPException(status_code=404, detail="Group not found")


@router.get("/")
def read_root(req: Request) -> H5Root:
    logging.debug("data %s", req.app.state.get_data()[0])
    data, _ = req.app.state.get_data()
    if isinstance(data, dict):
        uuid = _path_to_uuid([], collection_type="g-")
        ret = H5Root(root=uuid)
        return ret
    raise Exception("data object is not a dict")
