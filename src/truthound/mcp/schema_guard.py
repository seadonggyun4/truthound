"""Bounded in-document schema reachability, without resolving external resources."""

from urllib.parse import unquote

from truthound.mcp.policy import MCPError


def validate_json_schema(root):
    visited, anchors = set(), None

    def resolve(reference):
        nonlocal anchors
        fragment = unquote(reference[1:])
        if not fragment:
            return (root,)
        if fragment.startswith("/"):
            target = root
            try:
                for part in fragment[1:].split("/"):
                    part = part.replace("~1", "/").replace("~0", "~")
                    if isinstance(target, dict):
                        target = target[part]
                    elif isinstance(target, list) and part.isascii() and part.isdigit():
                        target = target[int(part)]
                    else:
                        raise KeyError(part)
            except (KeyError, IndexError, ValueError, OverflowError):
                raise MCPError("invalid_arguments") from None
            return (target,)
        if anchors is None:
            anchors, pending, count = {}, [root], 0
            while pending:
                node = pending.pop()
                count += 1
                if count > 10000:
                    raise MCPError("resource_exhausted")
                if isinstance(node, dict):
                    # Check every candidate for ambiguous anchors. Literal JSON
                    # alone is inert, but may become a schema through a ref.
                    for key in ("$id", "id", "$anchor", "$dynamicAnchor"):
                        value = node.get(key)
                        if isinstance(value, str):
                            name = unquote(value[1:]) if value.startswith("#") else value
                            anchors.setdefault(name, []).append(node)
                    pending.extend(node.values())
                elif isinstance(node, list):
                    pending.extend(node)
        candidates = anchors.get(fragment)
        if not candidates:
            raise MCPError("operator_configuration_required")
        return candidates

    def visit(node, depth=0):
        if not isinstance(node, dict) or id(node) in visited:
            return
        if depth > 32 or len(visited) >= 10000:
            raise MCPError("resource_exhausted")
        visited.add(id(node))
        if "pattern" in node or "patternProperties" in node:
            raise MCPError("operator_configuration_required")
        for key in ("$ref", "$dynamicRef", "$recursiveRef", "$id", "id"):
            if key in node and (not isinstance(node[key], str) or not node[key].startswith("#")):
                raise MCPError("operator_configuration_required")
        for key in ("$ref", "$dynamicRef", "$recursiveRef"):
            if key in node:
                for target in resolve(node[key]):
                    visit(target, depth + 1)
        for key in ("properties", "$defs", "definitions", "dependentSchemas"):
            children = node.get(key)
            if isinstance(children, dict):
                for child in children.values():
                    visit(child, depth + 1)
        for key in (
            "additionalProperties",
            "unevaluatedProperties",
            "additionalItems",
            "contains",
            "not",
            "propertyNames",
            "if",
            "then",
            "else",
            "items",
            "unevaluatedItems",
            "contentSchema",
        ):
            child = node.get(key)
            for item in child if isinstance(child, list) else (child,):
                visit(item, depth + 1)
        for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
            children = node.get(key)
            if isinstance(children, list):
                for child in children:
                    visit(child, depth + 1)
        children = node.get("dependencies")
        if isinstance(children, dict):
            for child in children.values():
                visit(child, depth + 1)

    visit(root)
