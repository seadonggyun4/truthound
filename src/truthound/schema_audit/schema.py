"""JSON Schema 2020-12 structural contract derived from the typed value model.

Cross-reference integrity and semantic invariants additionally require load_document.
"""

from __future__ import annotations

import types
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Literal, Union, get_args, get_origin

from ._contract import MAX_ITEMS, MAX_STRING, hints
from .serialization import ROOTS


def json_schema() -> dict:
    definitions = {}

    def node(annotation):
        origin, args = get_origin(annotation), get_args(annotation)
        if origin in (types.UnionType, Union):
            return {"anyOf": [node(a) for a in args]}
        if origin is Literal:
            return {"enum": list(args)}
        if origin is tuple:
            return {"type": "array", "items": node(args[0]), "maxItems": MAX_ITEMS}
        if isinstance(annotation, type) and issubclass(annotation, Enum):
            return {"type": "string", "enum": [e.value for e in annotation]}
        if is_dataclass(annotation):
            name = annotation.__name__
            if name not in definitions:
                definitions[name] = {}
                definitions[name] = {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [f.name for f in fields(annotation)],
                    "properties": {key: node(a) for key, a in hints(annotation).items()},
                }
            return {"$ref": "#/$defs/" + name}
        return {
            str: {"type": "string", "maxLength": MAX_STRING},
            int: {"type": "integer", "minimum": -9007199254740991, "maximum": 9007199254740991},
            bool: {"type": "boolean"},
            type(None): {"type": "null"},
        }[annotation]

    roots = [node(cls) for cls in ROOTS.values()]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:truthound:schema-audit:1",
        "oneOf": roots,
        "$defs": definitions,
    }
