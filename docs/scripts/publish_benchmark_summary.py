"""Publish a release benchmark artifact into the docs summary page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from truthound.benchmark import ParityResult, classify_release_blockers


def _link(label: str, target: str | None) -> str:
    if not target:
        return f"`{label}`"
    return f"[{label}]({target})"


def _artifact_link(base_url: str | None, filename: str) -> str:
    if not base_url:
        return filename
    return f"{base_url.rstrip('/')}/{filename}"


def render_docs_summary(
    result: ParityResult,
    *,
    artifact_base_url: str | None = None,
    env_manifest_name: str = "env-manifest.json",
) -> str:
    grouped: dict[str, dict[str, object]] = {}
    for observation in result.observations:
        grouped.setdefault(observation.workload_id, {})[observation.framework] = observation

    release_claim_ready = bool(result.metadata.get("release_claim_ready"))
    release_blockers = dict(result.metadata.get("release_blockers", classify_release_blockers(result)))
    blocker_categories = tuple(release_blockers.get("categories", ()))
    primary_blocker = release_blockers.get("primary")

    lines = [
        "# Latest Benchmark Summary",
        "",
        "## Status",
        "",
        (
            "The latest release-grade artifact set cleared the fixed-runner 3.0 GA benchmark gate."
            if release_claim_ready
            else "The latest recorded artifact set did not clear the full 3.0 GA benchmark gate."
        ),
        "",
        f"- Suite: `{result.suite_name}`",
        f"- Passed: `{'yes' if not result.has_blocking_failures else 'no'}`",
        f"- Official claim eligible: `{'yes' if release_claim_ready else 'no'}`",
    ]
    if primary_blocker:
        lines.append(f"- Primary blocker: `{primary_blocker}`")
    if blocker_categories:
        lines.append(f"- Blocker categories: `{', '.join(blocker_categories)}`")

    lines.extend(
        [
            "",
            "## Artifact Links",
            "",
            f"- {_link('release-ga.json', _artifact_link(artifact_base_url, 'release-ga.json'))}",
            f"- {_link('release-ga.md', _artifact_link(artifact_base_url, 'release-ga.md'))}",
            f"- {_link('release-ga.html', _artifact_link(artifact_base_url, 'release-ga.html'))}",
            f"- {_link(env_manifest_name, _artifact_link(artifact_base_url, env_manifest_name))}",
            "",
            "## Comparable Workloads",
            "",
            "| Workload | Truthound Warm (s) | GX Warm (s) | Speedup | Memory Ratio | Correctness |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for workload_id in sorted(grouped):
        truthound_observation = grouped[workload_id].get("truthound")
        gx_observation = grouped[workload_id].get("gx")
        truthound_warm = "n/a"
        gx_warm = "n/a"
        speedup = "n/a"
        memory_ratio = "n/a"
        correctness = "incomplete"
        if truthound_observation is not None:
            truthound_warm = f"{truthound_observation.warm_median_seconds:.6f}"
        if gx_observation is not None:
            gx_warm = f"{gx_observation.warm_median_seconds:.6f}"
        if truthound_observation is not None and gx_observation is not None:
            if truthound_observation.warm_median_seconds > 0 and gx_observation.warm_median_seconds > 0:
                speedup = (
                    f"{gx_observation.warm_median_seconds / truthound_observation.warm_median_seconds:.2f}x"
                )
            if truthound_observation.peak_rss_bytes > 0 and gx_observation.peak_rss_bytes > 0:
                memory_ratio = (
                    f"{truthound_observation.peak_rss_bytes / gx_observation.peak_rss_bytes:.2%}"
                )
            correctness = (
                "pass"
                if truthound_observation.correctness_passed and gx_observation.correctness_passed
                else "fail"
            )
        lines.append(
            "| "
            f"{workload_id} | "
            f"{truthound_warm} | "
            f"{gx_warm} | "
            f"{speedup} | "
            f"{memory_ratio} | "
            f"{correctness} |"
        )

    lines.extend(["", "## Assertions", ""])
    for assertion in result.assertions:
        status = "PASS" if assertion.passed else "FAIL"
        lines.append(f"- [{status}] `{assertion.name}`: {assertion.message}")

    lines.extend(
        [
            "",
            "## Related Reading",
            "",
            "- [Performance and Benchmarks](../guides/performance.md)",
            "- [Benchmark Methodology](../guides/benchmark-methodology.md)",
            "- [GX Parity Gate](../guides/gx-parity.md)",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="Path to release-ga.json")
    parser.add_argument(
        "--output",
        default="docs/releases/latest-benchmark-summary.md",
        help="Path to the docs summary page to write.",
    )
    parser.add_argument(
        "--artifact-base-url",
        default=None,
        help="Optional base URL or relative path prefix used to link the release artifacts.",
    )
    parser.add_argument(
        "--env-manifest-name",
        default="env-manifest.json",
        help="Filename to use for the environment-manifest link label.",
    )
    args = parser.parse_args()

    result_path = Path(args.json)
    result = ParityResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_docs_summary(
            result,
            artifact_base_url=args.artifact_base_url,
            env_manifest_name=args.env_manifest_name,
        ),
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
