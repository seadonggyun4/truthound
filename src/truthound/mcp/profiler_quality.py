"""Selected native quality scorers and rules over pinned Arrow input."""

from truthound.mcp.governance_worker import selected
from truthound.mcp.policy import MCPError


def execute(owner, args, inputs):
    import polars as pl

    owner.guard()
    if args["command"] in {"quality_history", "quality_trend"}:
        trend = selected(owner.runtime.quality_trends, args["trend_id"])
        alias = args["rule_ids"][0]
        rule = selected(owner.runtime.quality_rules, alias)
        if args["command"] == "quality_history":
            history = trend.get_history(rule.name, limit=args.get("limit", 50))
            result = {"rule_id": alias, "history": [
                {key: row[key] for key in ("timestamp", "metrics", "data_size")}
                for row in history
            ]}
        else:
            result = {"rule_id": alias, "trend": trend.analyze_trend(rule.name, args.get("window_days", 30))}
        owner.guard()
        return result
    scorer = selected(owner.runtime.quality_scorers, args["scorer_id"])
    if args["command"] == "quality_cache_clear":
        scorer.clear_cache()
        owner.guard()
        return {"cleared": True}
    if not inputs or set(inputs) != {"asset_path"}:
        raise MCPError("invalid_handle")
    frame = pl.read_ipc(inputs["asset_path"], memory_map=False)
    label_column = args.get("label_column")
    if label_column is not None and label_column not in frame.columns:
        raise MCPError("invalid_arguments")
    labels = frame.get_column(label_column) if label_column is not None else None
    results = []
    for alias in args["rule_ids"]:
        owner.guard()
        rule = selected(owner.runtime.quality_rules, alias)
        score = scorer.score(rule, frame, labels, use_cache=args.get("use_cache", True))
        owner.guard()
        if args["command"] == "quality_record":
            trend = selected(owner.runtime.quality_trends, args["trend_id"])
            trend.record(rule.name, score.metrics, data_size=frame.height)
            owner.guard()
        results.append({
            "rule_id": alias, "rule_type": score.rule_type.value, "column": score.column,
            "metrics": score.metrics.to_dict(), "should_use": score.should_use,
        })
    if args["command"] == "quality_compare":
        results.sort(key=lambda row: row["metrics"]["f1_score"], reverse=True)
    return {"scores": results, "labels_provided": labels is not None}
