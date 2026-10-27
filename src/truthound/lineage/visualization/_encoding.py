"""Data encoding shared by native lineage text and HTML renderers."""

import re

_PREFIX = "truthound_id_"
_RESERVED = {"end", "graph", "digraph", "subgraph", "node", "edge", "strict"}


def safe_node_id(value: str) -> str:
    """Preserve ordinary IDs and encode every other ID without collisions."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) and not (
        value.startswith(_PREFIX) or value.lower() in _RESERVED
    ):
        return value
    return _PREFIX + value.encode("utf-8").hex()


def javascript_json(value: str) -> str:
    """Keep JSON data from terminating its containing HTML script element."""
    for character, escaped in (
        ("&", r"\u0026"),
        ("<", r"\u003c"),
        (">", r"\u003e"),
        ("\u2028", r"\u2028"),
        ("\u2029", r"\u2029"),
    ):
        value = value.replace(character, escaped)
    return value


def mermaid_label(value: str) -> str:
    """Encode syntax characters using Mermaid's decimal entity notation."""
    special = set("\"'[]{}()|;#\\<>`&\r\n")
    return "".join(
        f"#{ord(character)};" if character in special else character for character in value
    )
