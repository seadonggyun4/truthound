"""Deterministic fixed layouts shared by native D3 data and its HTML viewer."""

from collections import deque
from math import ceil, cos, pi, sin, sqrt


def apply_d3_layout(data, config):
    data["config"]["orientation"] = config.orientation
    nodes = data["nodes"]
    count = len(nodes)
    if not count or config.layout == "force":
        return data
    if config.layout not in {"grid", "circular", "hierarchical"}:
        # JSON remains usable by clients with custom layout extensions.
        return data
    width, height = config.width, config.height
    if config.layout == "grid":
        columns = max(1, ceil(sqrt(count * width / height)))
        rows = ceil(count / columns)
        positions = [
            ((index % columns + 0.5) * width / columns, (index // columns + 0.5) * height / rows)
            for index in range(count)
        ]
    elif config.layout == "circular":
        radius = 0.4 * min(width, height)
        positions = [
            (
                width / 2 + radius * cos(2 * pi * index / count),
                height / 2 + radius * sin(2 * pi * index / count),
            )
            for index in range(count)
        ]
    else:
        incoming = [0] * count
        outgoing = [[] for _ in nodes]
        for link in data["links"]:
            source, target = link["source"], link["target"]
            outgoing[source].append(target)
            incoming[target] += 1
        levels = [0] * count
        pending = deque(index for index, degree in enumerate(incoming) if degree == 0)
        processed = set()
        while pending:
            source = pending.popleft()
            processed.add(source)
            for target in outgoing[source]:
                levels[target] = max(levels[target], levels[source] + 1)
                incoming[target] -= 1
                if incoming[target] == 0:
                    pending.append(target)
        # Cycles and nodes blocked by them share a final rank, without dropping nodes.
        final = max((levels[index] for index in processed), default=-1) + 1
        for index in range(count):
            if index not in processed:
                levels[index] = final
        groups = {}
        for index, level in enumerate(levels):
            groups.setdefault(level, []).append(index)
        ranks = max(groups) + 1
        positions = [None] * count
        horizontal = config.orientation in {"LR", "RL"}
        for level, indices in groups.items():
            for offset, index in enumerate(indices):
                along = (level + 0.5) / ranks
                across = (offset + 0.5) / len(indices)
                if config.orientation in {"BT", "RL"}:
                    along = 1 - along
                positions[index] = (
                    (width * along, height * across)
                    if horizontal
                    else (width * across, height * along)
                )
    for node, (x, y) in zip(nodes, positions, strict=True):
        node.update(x=x, y=y, fx=x, fy=y)
    return data
