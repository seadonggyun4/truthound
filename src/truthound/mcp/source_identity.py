"""Version-neutral semantic source identity for supported Python runtimes.

ast.dump output is not stable between Python 3.11/3.12/3.13 (for example empty
type_params and show_empty). Normalize fields explicitly, never platform paths,
line numbers, CRLF bytes, reprs or interpreter-specific AST dump formatting.
"""

import ast
import base64
import hashlib
import inspect
import json
import math
import textwrap


def _tree(value):
    if isinstance(value, ast.AST):
        return [
            type(value).__name__,
            {
                key: _tree(item)
                for key, item in ast.iter_fields(value)
                if not (key == "type_params" and not item)
            },
        ]
    if isinstance(value, list):
        return [_tree(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes": base64.b64encode(value).decode("ascii")}
    if value is Ellipsis:
        return {"ellipsis": True}
    if isinstance(value, complex):
        return {"complex": [_tree(value.real), _tree(value.imag)]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    return value


def source_identity(source):
    document = _tree(ast.parse(textwrap.dedent(source), feature_version=(3, 11)))
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def class_identity(cls):
    return source_identity(inspect.getsource(cls))
