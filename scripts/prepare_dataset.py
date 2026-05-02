from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PAIR_PATTERN = re.compile(r"(?P<trade_date>\d{4}-\d{2}-\d{2})_(?P<symbol>[A-Z0-9]{3,12}\.(?:SH|SZ))_order\.csv")
SPLIT_PREFIX = {"train": "TRN", "validation": "VAL", "test": "TST"}
SPLIT_BASE_DATE = {
    "train": pd.Timestamp("2000-01-01"),
    "validation": pd.Timestamp("2001-01-01"),
    "test": pd.Timestamp("2002-01-01"),
}
MANIFEST_COLUMNS = [
    "split",
    "sample_id",
    "trade_date",
    "symbol",
    "order_file",
    "trade_file",
    "prev_close_found",
]
PREV_CLOSE_COLUMNS = ["sample_id", "trade_date", "symbol", "prev_close"]


@dataclass
class Sample:
    order_path: Path
    trade_path: Path
    trade_date: str
    symbol: str


def find_task_root(source_root: Path) -> Path:
    for child in source_root.iterdir():
        if child.is_dir() and child.name == "开盘竞价任务文件":
            return child
    raise FileNotFoundError("没有在 source-root 下找到“开盘竞价任务文件”目录")


def collect_pairs(folder: Path) -> list[Sample]:
    samples: list[Sample] = []
    for order_path in sorted(folder.glob("*_order.csv")):
        match = PAIR_PATTERN.match(order_path.name)
        if not match:
            continue
        trade_path = order_path.with_name(order_path.name.replace("_order.csv", "_trade.csv"))
        if not trade_path.exists():
            continue
        samples.append(
            Sample(
                order_path=order_path,
                trade_path=trade_path,
                trade_date=match.group("trade_date"),
                symbol=match.group("symbol"),
            )
        )
    return samples


def read_prev_close_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path, encoding="utf-8-sig")

    column_map = {}
    for column in raw.columns:
        name = str(column)
        if name in {"sym", "股票代码"}:
            column_map[column] = "symbol"
        elif name in {"date", "交易日期"}:
            column_map[column] = "trade_date"
        elif name in {"prev_close", "前收盘价"}:
            column_map[column] = "prev_close"

    normalized = raw.rename(columns=column_map)
    normalized = normalized[[col for col in ["trade_date", "symbol", "prev_close"] if col in normalized.columns]].copy()
    if set(normalized.columns) != {"trade_date", "symbol", "prev_close"}:
        raise ValueError(f"前收盘价文件列名无法识别：{path}")

    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
    normalized["symbol"] = normalized["symbol"].astype(str)
    normalized["prev_close"] = normalized["prev_close"].astype(float)
    return normalized.drop_duplicates(["trade_date", "symbol"], keep="last")


def iter_prev_close_files(task_root: Path) -> list[Path]:
    paths = []
    for path in sorted(task_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx"}:
            continue
        name = path.name.lower()
        if "prev" in name and "close" in name:
            paths.append(path)
    return paths


def build_prev_close_table(task_root: Path) -> pd.DataFrame:
    frames = [read_prev_close_file(path) for path in iter_prev_close_files(task_root)]
    if not frames:
        raise FileNotFoundError("没有找到前收盘价文件")

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(["trade_date", "symbol"], keep="last").sort_values(["trade_date", "symbol"])


def discover_sample_groups(task_root: Path) -> list[list[Sample]]:
    groups = []
    for folder in sorted(task_root.iterdir()):
        if not folder.is_dir():
            continue
        samples = collect_pairs(folder)
        if samples:
            groups.append(samples)
    return groups


def anonymized_symbol(sample_id: str, original_symbol: str) -> str:
    exchange = original_symbol.split(".")[-1]
    number = sample_id[-3:]
    return f"ANON{number}.{exchange}"


def synthetic_trade_date(split: str, index: int) -> str:
    return (SPLIT_BASE_DATE[split] + pd.Timedelta(days=index - 1)).strftime("%Y-%m-%d")


def anonymize_datetime_column(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if "datetime" not in frame.columns:
        return frame

    result = frame.copy()
    parsed = pd.to_datetime(result["datetime"], errors="coerce")
    base_date = pd.Timestamp(trade_date)
    offsets = parsed - parsed.dt.normalize()
    anonymized = base_date + offsets
    result["datetime"] = anonymized.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str.rstrip("0").str.rstrip(".")
    return result


def write_anonymized_tick_file(source_path: Path, output_path: Path, trade_date: str, symbol: str) -> None:
    frame = pd.read_csv(source_path)
    if "sym" in frame.columns:
        frame["sym"] = symbol
    frame = anonymize_datetime_column(frame, trade_date)
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")


def copy_split(split: str, samples: list[Sample], prev_close: pd.DataFrame, data_root: Path) -> pd.DataFrame:
    split_root = data_root / split
    orders_dir = split_root / "orders"
    trades_dir = split_root / "trades"
    orders_dir.mkdir(parents=True, exist_ok=True)
    trades_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prev_rows = []
    prev_lookup = {
        (row.trade_date, row.symbol): row.prev_close
        for row in prev_close.itertuples(index=False)
    }

    for index, sample in enumerate(samples, start=1):
        sample_id = f"{SPLIT_PREFIX[split]}{index:03d}"
        anonymized_date = synthetic_trade_date(split, index)
        anonymized_code = anonymized_symbol(sample_id, sample.symbol)
        copied_order = f"{sample_id}_order.csv"
        copied_trade = f"{sample_id}_trade.csv"
        write_anonymized_tick_file(sample.order_path, orders_dir / copied_order, anonymized_date, anonymized_code)
        write_anonymized_tick_file(sample.trade_path, trades_dir / copied_trade, anonymized_date, anonymized_code)

        prev_value = prev_lookup.get((sample.trade_date, sample.symbol))
        prev_found = prev_value is not None and not pd.isna(prev_value)
        if prev_found:
            prev_rows.append(
                {
                    "sample_id": sample_id,
                    "trade_date": anonymized_date,
                    "symbol": anonymized_code,
                    "prev_close": float(prev_value),
                }
            )

        rows.append(
            {
                "split": split,
                "sample_id": sample_id,
                "trade_date": anonymized_date,
                "symbol": anonymized_code,
                "order_file": f"orders/{copied_order}",
                "trade_file": f"trades/{copied_trade}",
                "prev_close_found": prev_found,
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    split_prev = pd.DataFrame(prev_rows, columns=PREV_CLOSE_COLUMNS).drop_duplicates(["sample_id"], keep="last")
    manifest.to_csv(split_root / "manifest.csv", index=False, encoding="utf-8-sig")
    split_prev.to_csv(split_root / "prev_close.csv", index=False, encoding="utf-8-sig")
    return manifest


def prepare_dataset(source_root: Path, project_root: Path) -> None:
    task_root = find_task_root(source_root)
    data_root = project_root / "data"
    if data_root.exists():
        shutil.rmtree(data_root)
    (data_root / "manifests").mkdir(parents=True, exist_ok=True)

    sample_groups = discover_sample_groups(task_root)
    if not sample_groups:
        raise ValueError("没有找到逐笔委托/成交样本")
    official_samples = next((samples for samples in sample_groups if len(samples) == 20), None)
    if official_samples is None and len(sample_groups) == 1:
        official_samples = sample_groups[0]
    if official_samples is None:
        raise ValueError("没有找到包含 20 个样本的开发/验证数据目录")
    train_samples = official_samples[:14]
    validation_samples = official_samples[14:]

    test_samples: list[Sample] = []
    for samples in sample_groups:
        if samples is not official_samples:
            test_samples.extend(samples)

    prev_close = build_prev_close_table(task_root)
    manifests = [
        copy_split("train", train_samples, prev_close, data_root),
        copy_split("validation", validation_samples, prev_close, data_root),
        copy_split("test", test_samples, prev_close, data_root),
    ]
    all_samples = pd.concat(manifests, ignore_index=True)
    all_samples.to_csv(data_root / "manifests" / "all_samples.csv", index=False, encoding="utf-8-sig")

    print(f"train 样本数: {len(train_samples)}")
    print(f"validation 样本数: {len(validation_samples)}")
    print(f"test 样本数: {len(test_samples)}")
    print(f"总样本数: {len(all_samples)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 train / validation / test 三份开盘竞价数据。")
    parser.add_argument("--source-root", type=Path, default=Path(".."), help="原始项目根目录，默认是当前仓库的父目录。")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    prepare_dataset(args.source_root.resolve(), project_root)


if __name__ == "__main__":
    main()
