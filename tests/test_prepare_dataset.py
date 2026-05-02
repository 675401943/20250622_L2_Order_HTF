from pathlib import Path

import pandas as pd

from scripts.prepare_dataset import prepare_dataset


def test_prepare_dataset_anonymizes_file_names_manifest_and_csv_content(tmp_path: Path):
    source_root = tmp_path / "source"
    project_root = tmp_path / "project"
    task_root = source_root / "开盘竞价任务文件"
    data_folder = task_root / "source_group_a"
    data_folder.mkdir(parents=True)
    project_root.mkdir()

    order_file = data_folder / "2099-01-01_SRC001.SZ_order.csv"
    trade_file = data_folder / "2099-01-01_SRC001.SZ_trade.csv"
    pd.DataFrame(
        [
            {
                "datetime": "2099-01-01 09:15:00.010",
                "sym": "SRC001.SZ",
                "price": 10.01,
                "size": 100,
                "side": 1,
                "orderid": 1,
            }
        ]
    ).to_csv(order_file, index=False)
    pd.DataFrame(
        [
            {
                "datetime": "2099-01-01 09:30:00.010",
                "sym": "SRC001.SZ",
                "price": 10.01,
                "size": 100,
                "bidorderid": 1,
                "askorderid": 2,
            }
        ]
    ).to_csv(trade_file, index=False)
    pd.DataFrame(
        [{"sym": "SRC001.SZ", "date": 20990101, "prev_close": 10.00}]
    ).to_excel(task_root / "prevclose_20.xlsx", index=False)

    prepare_dataset(source_root, project_root)

    manifest = pd.read_csv(project_root / "data" / "train" / "manifest.csv")
    assert manifest.loc[0, "sample_id"] == "TRN001"
    assert manifest.loc[0, "symbol"] == "ANON001.SZ"
    assert manifest.loc[0, "trade_date"] == "2000-01-01"
    assert manifest.loc[0, "order_file"] == "orders/TRN001_order.csv"
    assert "SRC001" not in manifest.to_csv(index=False)
    assert "2099-01-01" not in manifest.to_csv(index=False)

    order_text = (project_root / "data" / "train" / "orders" / "TRN001_order.csv").read_text()
    trade_text = (project_root / "data" / "train" / "trades" / "TRN001_trade.csv").read_text()
    assert "SRC001.SZ" not in order_text
    assert "SRC001.SZ" not in trade_text
    assert "2099-01-01" not in order_text
    assert "2099-01-01" not in trade_text
    assert "ANON001.SZ" in order_text
    assert "2000-01-01" in trade_text
