from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


def _extract_nodeids(path: Path) -> list[str]:
    nodeids: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("tests/") or line.startswith("tests\\"):
            nodeids.append(line)
    return nodeids


def _ensure_unique(nodeids: Iterable[str], *, label: str) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []

    for nodeid in nodeids:
        if nodeid in seen:
            duplicates.append(nodeid)
            continue
        seen.add(nodeid)
        unique.append(nodeid)

    if duplicates:
        duplicate_preview = ", ".join(sorted(set(duplicates))[:5])
        raise ValueError(f"{label} contains duplicate node ids: {duplicate_preview}")

    return unique


def _group_by_file(nodeids: Iterable[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for nodeid in nodeids:
        grouped[nodeid.split("::", 1)[0]].append(nodeid)
    return dict(grouped)


def _dedupe_lane_overlap(
    contract_nodeids: list[str],
    fault_e2e_nodeids: list[str],
) -> tuple[list[str], list[str], list[str]]:
    fault_lookup = set(fault_e2e_nodeids)
    overlap = [nodeid for nodeid in contract_nodeids if nodeid in fault_lookup]
    if not overlap:
        return contract_nodeids, fault_e2e_nodeids, []

    overlap_lookup = set(overlap)
    deduped_contract = [nodeid for nodeid in contract_nodeids if nodeid not in overlap_lookup]
    if not deduped_contract:
        raise ValueError("contract selection would become empty after removing overlapping node ids")

    return deduped_contract, fault_e2e_nodeids, overlap


@dataclass
class ContractShard:
    shard_id: int
    nodeids: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        return len(self.nodeids)


def _build_contract_shards(
    contract_nodeids: list[str],
    *,
    shard_count: int,
) -> list[ContractShard]:
    if shard_count < 1:
        raise ValueError("contract shard count must be at least 1")
    if not contract_nodeids:
        raise ValueError("contract selection is empty")

    grouped = _group_by_file(contract_nodeids)
    if shard_count > len(grouped):
        raise ValueError(
            f"cannot build {shard_count} non-empty contract shards from only {len(grouped)} files"
        )

    shards = [ContractShard(shard_id=index) for index in range(shard_count)]
    for file_path, nodeids in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(shards, key=lambda shard: (shard.node_count, shard.shard_id))
        target.files.append(file_path)
        target.nodeids.extend(nodeids)

    if any(not shard.nodeids for shard in shards):
        raise ValueError("generated at least one empty contract shard")

    assigned = [nodeid for shard in shards for nodeid in shard.nodeids]
    if set(assigned) != set(contract_nodeids) or len(assigned) != len(contract_nodeids):
        raise ValueError("contract shards do not cover the collected node ids exactly once")

    return shards


def _write_manifest(path: Path, nodeids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(nodeids)
    path.write_text(f"{content}\n" if content else "", encoding="utf-8")


def _build_summary(
    *,
    contract_nodeids: list[str],
    fault_e2e_nodeids: list[str],
    e2e_nodeids: list[str],
    contract_shards: list[ContractShard],
    overlap_nodeids: list[str],
) -> dict[str, object]:
    return {
        "total_selected": len(contract_nodeids) + len(fault_e2e_nodeids),
        "contract_selected": len(contract_nodeids),
        "fault_e2e_selected": len(fault_e2e_nodeids),
        "e2e_selected": len(e2e_nodeids),
        "overlap_selected": len(overlap_nodeids),
        "contract_shard_count": len(contract_shards),
        "contract_shards": [
            {
                "shard_id": shard.shard_id,
                "manifest": f"contract-{shard.shard_id}.txt",
                "node_count": shard.node_count,
                "file_count": len(shard.files),
            }
            for shard in contract_shards
        ],
        "fault_e2e_manifest": {
            "manifest": "fault-e2e.txt",
            "node_count": len(fault_e2e_nodeids),
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build balanced pytest shard manifests for the PR quality gate.",
    )
    parser.add_argument("--contract-nodeids", type=Path, required=True)
    parser.add_argument("--fault-nodeids", type=Path, required=True)
    parser.add_argument("--e2e-nodeids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--contract-shards", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        contract_nodeids = _ensure_unique(
            _extract_nodeids(args.contract_nodeids),
            label="contract collect output",
        )
        fault_e2e_nodeids = _ensure_unique(
            _extract_nodeids(args.fault_nodeids),
            label="fault/e2e collect output",
        )
        e2e_nodeids = _ensure_unique(
            _extract_nodeids(args.e2e_nodeids) if args.e2e_nodeids else [],
            label="e2e collect output",
        )

        if not set(e2e_nodeids).issubset(fault_e2e_nodeids):
            raise ValueError("e2e collect output must be a subset of the fault/e2e selection")

        contract_nodeids, fault_e2e_nodeids, overlap_nodeids = _dedupe_lane_overlap(
            contract_nodeids,
            fault_e2e_nodeids,
        )

        contract_shards = _build_contract_shards(
            contract_nodeids,
            shard_count=args.contract_shards,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for shard in contract_shards:
            _write_manifest(args.output_dir / f"contract-{shard.shard_id}.txt", shard.nodeids)
        _write_manifest(args.output_dir / "fault-e2e.txt", fault_e2e_nodeids)

        summary = _build_summary(
            contract_nodeids=contract_nodeids,
            fault_e2e_nodeids=fault_e2e_nodeids,
            e2e_nodeids=e2e_nodeids,
            contract_shards=contract_shards,
            overlap_nodeids=overlap_nodeids,
        )
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        print("Quality shard summary:")
        print(f"  total selected: {summary['total_selected']}")
        print(f"  contract selected: {summary['contract_selected']}")
        print(f"  fault/e2e selected: {summary['fault_e2e_selected']}")
        print(f"  e2e selected: {summary['e2e_selected']}")
        if overlap_nodeids:
            print(f"  overlap selected: {summary['overlap_selected']} (kept in fault/e2e lane)")
        for shard in contract_shards:
            print(
                f"  contract shard {shard.shard_id}: "
                f"{shard.node_count} tests across {len(shard.files)} files"
            )
        if not e2e_nodeids:
            print("  info: e2e selected count is 0 (non-blocking)")
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
