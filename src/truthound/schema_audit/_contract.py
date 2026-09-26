"""Strict, dependency-free value construction shared by the JSON boundary."""

from __future__ import annotations

import re
import types
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
from typing import Literal, Union, get_args, get_origin, get_type_hints

VERSION = "truthound.schema-audit/1"
MAX_BYTES = 20 * 1024 * 1024
MAX_DEPTH = 32
MAX_STRING = 4096
MAX_ITEMS = 50000
MAX_NODES = 1000000


class ModelError(ValueError):
    """A stable code and structural JSON pointer, never an input value."""

    def __init__(self, code: str, pointer: str = "") -> None:
        self.code = code
        self.pointer = pointer
        super().__init__(f"{code} at {pointer or '/'}")


def require(condition: bool, code: str, pointer: str = "") -> None:
    if not condition:
        raise ModelError(code, pointer)


def text(value: str, pointer: str) -> None:
    require(len(value) <= MAX_STRING, "string_limit", pointer)
    require(
        not any(0xD800 <= ord(c) <= 0xDFFF or c == "\x00" for c in value),
        "invalid_unicode",
        pointer,
    )


def opaque(value: str, pointer: str) -> None:
    require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value)), "invalid_id", pointer)


@lru_cache(maxsize=64)
def hints(cls: type) -> dict:
    return get_type_hints(cls)


def convert(value, annotation, pointer: str, *, wire: bool, max_items: int = MAX_ITEMS):
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (types.UnionType, Union):
        for option in args:
            try:
                return convert(value, option, pointer, wire=wire)
            except ModelError:
                pass
        raise ModelError("invalid_type", pointer)
    if origin is Literal:
        require(
            any(type(value) is type(x) and value == x for x in args), "invalid_literal", pointer
        )
        return value
    if origin is tuple:
        require(type(value) is (list if wire else tuple), "immutable_array_required", pointer)
        require(len(value) <= max_items, "item_limit", pointer)
        return tuple(convert(v, args[0], f"{pointer}/{i}", wire=wire) for i, v in enumerate(value))
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if wire:
            require(type(value) is str, "invalid_enum", pointer)
            try:
                return annotation(value)
            except ValueError:
                raise ModelError("invalid_enum", pointer) from None
        require(type(value) is annotation, "invalid_enum", pointer)
        return value
    if is_dataclass(annotation):
        if not wire:
            require(type(value) is annotation, "invalid_model", pointer)
            return value
        require(type(value) is dict, "object_required", pointer)
        known = hints(annotation)
        require(not (value.keys() - known.keys()), "unknown_field", pointer)
        required = {f.name for f in fields(annotation)}
        require(required <= value.keys(), "missing_field", pointer)
        limits = {f.name: f.metadata.get("max_items", MAX_ITEMS) for f in fields(annotation)}
        kwargs = {
            k: convert(v, known[k], f"{pointer}/{k}", wire=True, max_items=limits[k])
            for k, v in value.items()
        }
        try:
            return annotation(**kwargs)
        except ModelError as exc:
            raise ModelError(exc.code, pointer + exc.pointer) from None
    require(type(value) is annotation, "invalid_type", pointer)
    if annotation is str:
        text(value, pointer)
    elif annotation is int:
        require(abs(value) <= 9007199254740991, "integer_limit", pointer)
    return value


class Contract:
    """Frozen subclasses also reject mutable or coerced Python constructor inputs."""

    def __post_init__(self) -> None:
        limits = {f.name: f.metadata.get("max_items", MAX_ITEMS) for f in fields(self)}
        for key, annotation in hints(type(self)).items():
            convert(getattr(self, key), annotation, "/" + key, wire=False, max_items=limits[key])
        self._validate()

    def _validate(self) -> None:
        pass


def unique_ids(items: tuple, pointer: str) -> dict:
    index = {}
    for i, item in enumerate(items):
        opaque(item.id, f"{pointer}/{i}/id")
        require(item.id not in index, "duplicate_id", f"{pointer}/{i}/id")
        index[item.id] = item
    return index
