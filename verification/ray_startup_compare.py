"""Fixed paired real-SDK/Core startup control; never retries a failed pair."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from native_distributed_runner import OwnedProbe


def trial(mode, log):
    from truthound.mcp.tables import decode

    runtime = None
    failure = None
    diagnostic = {}
    started = time.monotonic()
    try:
        runtime = OwnedProbe("ray", log, native_ray_startup=mode == "sdk")
        ready = runtime.receive()
        if ready.get("ready") is not True:
            failure = "not_ready"
        elif "ray-start-" + mode not in runtime._diagnostic_report()["phases"]:
            failure = "startup_mode"
        else:
            value = decode(runtime.call("basics"))
            if value["rows"] != 3 or value["stats"]["count"] != 2 or value["stats"]["mean"] != 2:
                failure = "native_computation"
    except TimeoutError:
        failure = "timeout"
    except Exception:
        failure = "protocol_or_sdk_failure"
    finally:
        if runtime is not None:
            diagnostic = runtime._diagnostic_report()
            try:
                runtime.close()
            except Exception:
                failure = "cleanup_failure"
            diagnostic["driver_exit_after_cleanup"] = runtime.process.poll()
            if not failure and runtime.process.returncode != 0:
                failure = "driver_exit"
    return {"mode": mode, "failure": failure,
            "elapsed_seconds": round(time.monotonic() - started, 3), "diagnostic": diagnostic}


def save(output, results, pairs):
    complete = len(results) == pairs * 2
    report = {"pairs_planned": pairs, "trials_completed": len(results), "complete": complete,
              "passed": complete and all(row["failure"] is None for row in results),
              "reply_deadline_seconds": 30, "results": results}
    output.mkdir(parents=True, exist_ok=True)
    (output / "ray-startup-compare.json").write_text(json.dumps(report, indent=2) + "\n")
    suite = ET.Element("testsuite", name="ray-startup-compare", tests=str(len(results)),
                       failures=str(sum(row["failure"] is not None for row in results)))
    for index, row in enumerate(results):
        case = ET.SubElement(suite, "testcase", name=f"pair-{index // 2 + 1}-{row['mode']}",
                             classname="native_ray_startup", time=str(row["elapsed_seconds"]))
        properties = ET.SubElement(case, "properties")
        ET.SubElement(properties, "property", name="native_startup",
                      value=json.dumps(row["diagnostic"], sort_keys=True))
        if row["failure"]:
            ET.SubElement(case, "failure", message=row["failure"]).text = json.dumps(row["diagnostic"])
    ET.ElementTree(suite).write(output / "windows-native-ray-compare.xml", encoding="utf-8", xml_declaration=True)
    return report["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.pairs <= 20:
        parser.error("Expected between one and twenty fixed pairs")
    results = []
    with tempfile.TemporaryDirectory(prefix="truthound-ray-compare-") as raw:
        for index in range(args.pairs):
            order = ("core", "sdk") if index % 2 == 0 else ("sdk", "core")
            pair = []
            for mode in order:
                result = trial(mode, Path(raw) / f"{index}-{mode}.log")
                pair.append(result)
                results.append(result)
                print(json.dumps({"pair": index + 1, "mode": mode, "failure": result["failure"],
                                  "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
            save(args.output, results, args.pairs)
            if any(row["failure"] for row in pair):
                return 1
    return 0 if save(args.output, results, args.pairs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
