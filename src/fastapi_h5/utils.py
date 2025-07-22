import base64
import copy
import logging
from typing import Any, Literal

import numpy as np

from fastapi_h5.h5types import (
    H5UUID,
    H5Attribute,
    H5CompType,
    H5Dataset,
    H5FloatType,
    H5IntType,
    H5Link,
    H5NamedType,
    H5ScalarShape,
    H5Shape,
    H5SimpleShape,
    H5StrType,
    H5Type,
    H5ValuedAttribute,
)

logger = logging.getLogger()


def _path_to_uuid(path: list[str], collection_type: Literal["g-", "d-"]) -> H5UUID:
    abspath = "/" + "/".join(path)
    logger.debug("abspath %s", abspath)
    uuid = base64.b16encode(abspath.encode()).decode()
    return H5UUID(collection_type + "h5dict-" + uuid)


def _get_obj_at_path(
    data: dict[str, Any], path: str | list[str]
) -> tuple[Any, list[str]]:
    obj = data
    clean_path = []
    if isinstance(path, str):
        path = path.split("/")
    for p in path:
        logger.debug("path part is: %s", p)
        if p == "":
            continue
        logger.debug("traverse %s", obj)
        obj = obj[p]
        clean_path.append(p)
    return obj, clean_path


def _uuid_to_obj(data: dict[str, Any], uuid: str) -> tuple[Any, list[str]]:
    logger.debug("parse %s", uuid)
    col_type, idstr, path = uuid.split("-")
    path = base64.b16decode(path).decode()
    logger.debug("raw path %s", path)
    return _get_obj_at_path(data, path)


def _canonical_to_h5(canonical: str) -> H5Type | None:
    if canonical.startswith(">"):
        order = "BE"
    else:
        order = "LE"
    bytelen = int(canonical[2:])
    htyp: H5Type | None = None
    if canonical[1] == "f":
        # floating
        htyp = H5FloatType(base=f"H5T_IEEE_F{8 * bytelen}{order}")
    elif canonical[1] in ["u", "i"]:
        signed = canonical[1].upper()
        htyp = H5IntType(base=f"H5T_STD_{signed}{8 * bytelen}{order}")
    else:
        logger.error("numpy type %s not available", canonical)

    return htyp


def _make_shape_type(obj: Any) -> tuple[H5Shape | None, H5Type | None]:
    h5shape: H5Shape | None = None
    h5type: H5Type | None = None

    if callable(obj):
        obj = obj()

    if isinstance(obj, int):
        h5shape = H5ScalarShape()
        h5type = H5IntType(base="H5T_STD_I64LE")
    elif isinstance(obj, float):
        h5shape = H5ScalarShape()
        h5type = H5FloatType(base="H5T_IEEE_F64LE")

    elif isinstance(obj, str):
        h5shape = H5ScalarShape()
        h5type = H5StrType()

    elif isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], str):
        h5shape = H5SimpleShape(dims=[len(obj)])
        h5type = H5StrType()
    else:
        try:
            arr = np.array(obj)
            h5shape = H5SimpleShape(dims=list(arr.shape))

            if len(arr.dtype.descr) > 1:
                # compound datatype
                fields = []
                for field in arr.dtype.descr:
                    typ = _canonical_to_h5(field[1])
                    if typ is not None:
                        fields.append(H5NamedType(name=field[0], type=typ))
                h5type = H5CompType(fields=fields)
            else:
                canonical = arr.dtype.descr[0][1]
                h5type = _canonical_to_h5(canonical)
            logging.debug("convert np dtype %s  to %s", arr.dtype, h5type)

        except Exception as e:
            logger.error("exception in handling code %s", e.__repr__())

    return h5shape, h5type


def _dataset_from_obj(
    data: Any, path: list[str], obj: Any, uuid: H5UUID
) -> H5Dataset | None:
    shape, typ = _make_shape_type(obj)
    if shape is not None and typ is not None:
        ret = H5Dataset(
            id=uuid,
            attributeCount=len(_get_obj_attrs(data, path)),
            shape=shape,
            type=typ,
        )
        return ret
    return None


def _get_group_link(obj: Any, path: list[str]) -> H5Link:
    collection: Literal["datasets", "groups"]
    collection_type: Literal["g-", "d-"]
    if isinstance(obj, dict):
        collection = "groups"
        collection_type = "g-"
    else:
        collection = "datasets"
        collection_type = "d-"
    return H5Link(
        collection=collection,
        id=_path_to_uuid(path, collection_type=collection_type),
        title=path[-1],
    )


def _get_group_links(obj: Any, path: list[str]) -> list[H5Link]:
    links = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if not isinstance(key, str):
                logger.warning("unable to use non-string key: %s as group name", key)
                continue
            if key.endswith("_attrs"):
                continue
            link = _get_group_link(val, path + [key])
            links.append(link)
    return links


def _get_attr(
    aobj: dict[str, Any], name: str, include_values: bool = True
) -> H5Attribute | H5ValuedAttribute | None:
    logger.debug("get attribute")
    if name in aobj:
        h5shape, h5type = _make_shape_type(aobj[name])
        logger.debug("ret shape %s type %s", h5shape, h5type)
        if h5shape is not None and h5type is not None:
            ret = H5Attribute(name=name, shape=h5shape, type=h5type)
            if include_values:
                val = copy.copy(aobj[name])
                if isinstance(val, np.ndarray):
                    val = val.tolist()
                ret = H5ValuedAttribute(**ret.model_dump(by_alias=True), value=val)
            return ret
    return None


def _make_attrs(
    aobj: dict[str, Any], include_values: bool = False
) -> list[H5Attribute] | list[H5ValuedAttribute]:
    logger.debug("make attributes of %s", aobj)
    attrs = []
    if isinstance(aobj, dict):
        for key, value in aobj.items():
            if not isinstance(key, str):
                logger.warning(
                    "unable to use non-string key: %s as attribute name", key
                )
                continue
            attr = _get_attr(aobj, key, include_values=include_values)
            if attr is not None:
                attrs.append(attr)
    return attrs


def _get_obj_attrs(
    data: dict[str, Any], path: list[str], include_values: bool = False
) -> list[H5Attribute] | list[H5ValuedAttribute]:
    if len(path) == 0:
        # get root attributes
        if "_attrs" in data:
            return _make_attrs(data["_attrs"], include_values=include_values)
    else:
        logger.debug("normal attr fetch")
        parent, _ = _get_obj_at_path(data, path[:-1])
        logger.debug("parent is %s", parent)
        if f"{path[-1]}_attrs" in parent:
            logger.debug("make attrs for")
            return _make_attrs(
                parent[f"{path[-1]}_attrs"], include_values=include_values
            )
    return []
