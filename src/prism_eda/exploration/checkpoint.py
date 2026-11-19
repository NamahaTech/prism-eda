"""Turn exploration state into bytes and back.

Checkpoints are plain JSON, not pickles: they are stored by the calling
application (often in a database), and loading one must never be able to run
code. Dataclasses are rebuilt from their type hints.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from typing import Any, TypeVar, Union

from prism_eda._serialization import to_jsonable

_T = TypeVar("_T")
FORMAT = 1


def dumps(value: Any) -> bytes:
    payload = {"format": FORMAT, "value": to_jsonable(value)}
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def loads(cls: type[_T], payload: bytes) -> _T | None:
    """Rebuild ``cls`` from a checkpoint; ``None`` if it cannot be used.

    An unreadable or older-format checkpoint is not an error — the work is
    simply done again.
    """
    try:
        document = json.loads(payload.decode("utf-8"))
        if document.get("format") != FORMAT:
            return None
        return typing.cast(_T, _revive(cls, document["value"]))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _revive(annotation: Any, value: Any) -> Any:
    if value is None or annotation is Any:
        return value
    origin = typing.get_origin(annotation)
    arguments = typing.get_args(annotation)
    if origin in (Union, types.UnionType):
        for option in arguments:
            if option is type(None):
                continue
            return _revive(option, value)
        return value
    if origin is tuple:
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(_revive(arguments[0], item) for item in value)
        if arguments:
            return tuple(
                _revive(kind, item)
                for kind, item in zip(arguments, value, strict=False)
            )
        return tuple(value)
    if origin is list:
        kind = arguments[0] if arguments else Any
        return [_revive(kind, item) for item in value]
    if origin is dict:
        kind = arguments[1] if len(arguments) == 2 else Any
        return {key: _revive(kind, item) for key, item in value.items()}
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        hints = typing.get_type_hints(annotation)
        known = {item.name for item in dataclasses.fields(annotation)}
        return annotation(
            **{
                name: _revive(hints.get(name, Any), item)
                for name, item in value.items()
                if name in known
            }
        )
    return value
