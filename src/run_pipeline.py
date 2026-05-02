from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .auction_matcher import calculate_open_price_for_files
except ImportError:
    from auction_matcher import calculate_open_price_for_files


SPLITS = ("train", "validation", "test")


def load_prev_close(split_root: Path) -> dict[tuple[str, str], float]:
    path = split_root / "prev_close.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    return {
        (str(row.trade_date), str(row.symbol)): float(row.prev_close)
        for row in df.itertuples(index=False)
    }


def run_one_split(project_root: Path, split: str) -> pd.DataFrame:
    split_root = project_root / "data" / split
    manifest_path = split_root / "manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    prev_lookup = load_prev_close(split_root)

    rows = []
    for sample in manifest.itertuples(index=False):
        prev_close = prev_lookup.get((str(sample.trade_date), str(sample.symbol)))
        result = calculate_open_price_for_files(
            order_file=split_root / str(sample.order_file),
            trade_file=split_root / str(sample.trade_file),
            prev_close=prev_close,
            stock_code=str(sample.symbol),
            trade_date=str(sample.trade_date),
        ).to_dict()
        result.update(
            {
                "split": split,
                "sample_id": sample.sample_id,
                "order_file": sample.order_file,
                "trade_file": sample.trade_file,
                "prev_close_found": bool(sample.prev_close_found),
            }
        )
        rows.append(result)

    result_df = pd.DataFrame(rows)
    results_root = project_root / "results"
    results_root.mkdir(exist_ok=True)
    result_df.to_csv(results_root / f"{split}_results.csv", index=False, encoding="utf-8-sig")
    return result_df


def summarize_split(df: pd.DataFrame, split: str) -> dict:
    successful = df[df["status"] == "success"].copy()
    matched = df[df["is_match"] == True].copy()
    errors = pd.to_numeric(successful["error"], errors="coerce").dropna()
    return {
        "split": split,
        "sample_count": len(df),
        "success_count": len(successful),
        "match_count": len(matched),
        "accuracy": len(matched) / len(df) if len(df) else 0.0,
        "avg_error": errors.mean() if not errors.empty else None,
        "max_error": errors.max() if not errors.empty else None,
    }


def format_float(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.{digits}f}"


def write_summary(project_root: Path, split_results: dict[str, pd.DataFrame]) -> None:
    summaries = [summarize_split(df, split) for split, df in split_results.items()]
    lines = [
        "# 开盘集合竞价撮合运行结果汇总",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 总览",
        "",
        "| 数据集 | 样本数 | 成功撮合 | 命中实际开盘价 | 准确率 | 平均误差 | 最大误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {split} | {sample_count} | {success_count} | {match_count} | {accuracy:.2%} | {avg_error} | {max_error} |".format(
                split=item["split"],
                sample_count=item["sample_count"],
                success_count=item["success_count"],
                match_count=item["match_count"],
                accuracy=item["accuracy"],
                avg_error=format_float(item["avg_error"]),
                max_error=format_float(item["max_error"]),
            )
        )

    all_results = pd.concat(split_results.values(), ignore_index=True)
    failed = all_results[all_results["status"] != "success"]
    mismatched = all_results[(all_results["status"] == "success") & (all_results["is_match"] == False)]

    lines.extend(
        [
            "",
            "## 规则说明",
            "",
            "本项目把交易所规则中的“未成交量最小”落到代码里时，使用的是买卖累计申报数量差最小：",
            "",
            "```text",
            "diff = abs(候选价以上买入累计量 - 候选价以下卖出累计量)",
            "```",
            "",
            "这也是 `results/*_results.csv` 中 `selected_rule` 和候选价计算的核心依据。",
            "",
            "## 未成功撮合样本",
            "",
        ]
    )
    if failed.empty:
        lines.append("无。")
    else:
        lines.append("| 数据集 | 样本编号 | 日期 | 匿名代码 | 原因 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in failed.head(30).itertuples(index=False):
            lines.append(f"| {row.split} | {row.sample_id} | {row.trade_date} | {row.symbol} | {row.message} |")
        if len(failed) > 30:
            lines.append(f"\n仅展示前 30 条，完整记录见对应 results CSV。")

    lines.extend(["", "## 未命中样本", ""])
    if mismatched.empty:
        lines.append("无。")
    else:
        lines.append("| 数据集 | 样本编号 | 日期 | 匿名代码 | 计算开盘价 | 实际开盘价 | 误差 |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: |")
        for row in mismatched.head(30).itertuples(index=False):
            lines.append(
                f"| {row.split} | {row.sample_id} | {row.trade_date} | {row.symbol} | "
                f"{format_float(row.calculated_open_price, 2)} | {format_float(row.actual_open_price, 2)} | {format_float(row.error)} |"
            )
        if len(mismatched) > 30:
            lines.append(f"\n仅展示前 30 条，完整记录见对应 results CSV。")

    (project_root / "results" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行开盘集合竞价撮合 pipeline。")
    parser.add_argument("--split", choices=("all", *SPLITS), default="all", help="选择要运行的数据集。")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    selected_splits = SPLITS if args.split == "all" else (args.split,)
    split_results = {split: run_one_split(project_root, split) for split in selected_splits}
    write_summary(project_root, split_results)
    for split, df in split_results.items():
        matched = int(df["is_match"].sum())
        print(f"{split}: {matched}/{len(df)} 命中")
    print("结果已写入 results/ 目录。")


if __name__ == "__main__":
    main()
