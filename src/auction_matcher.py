from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


AUCTION_ORDER_CUTOFF = pd.to_datetime("09:26:00").time()
ACTUAL_OPEN_TIME = pd.to_datetime("09:25:00").time()
CONTINUOUS_OPEN_TIME = pd.to_datetime("09:30:00").time()


@dataclass
class AuctionResult:
    trade_date: str
    symbol: str
    prev_close: Optional[float]
    calculated_open_price: Optional[float]
    actual_open_price: Optional[float]
    continuous_first_price: Optional[float]
    error: Optional[float]
    is_match: bool
    status: str
    message: str
    candidate_count: int = 0
    selected_rule: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _as_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def _round_price(value: float, decimals: int = 2) -> float:
    scale = 10**decimals
    return int(float(value) * scale + 0.5) / scale


def scaled_mean_round(prices: pd.Series, decimals: int = 2) -> float:
    values = [Decimal(str(price)) for price in prices]
    mean_value = sum(values) / Decimal(len(values))
    quant = Decimal("1").scaleb(-decimals)
    return float(mean_value.quantize(quant, rounding=ROUND_HALF_UP))


def generate_price_range(min_price: float, max_price: float, step: float = 0.01) -> np.ndarray:
    start = int(round(float(min_price) / step))
    end = int(round(float(max_price) / step))
    return np.round(np.arange(start, end + 1) * step, 2)


def remove_cancelled_orders(df_order: pd.DataFrame, df_trade: pd.DataFrame) -> pd.DataFrame:
    """删除 09:26 前已撤销的委托。"""
    if df_trade is None or df_trade.empty:
        return df_order.copy().reset_index(drop=True)

    trade = df_trade.copy()
    trade["datetime"] = _as_datetime(trade["datetime"])
    trade = trade[trade["datetime"].dt.time < AUCTION_ORDER_CUTOFF]

    cancelled = trade[
        ((trade["bidorderid"] > 0) & (trade["askorderid"] == 0))
        | ((trade["askorderid"] > 0) & (trade["bidorderid"] == 0))
    ]
    cancelled_ids = pd.concat(
        [
            cancelled.loc[cancelled["bidorderid"] > 0, "bidorderid"],
            cancelled.loc[cancelled["askorderid"] > 0, "askorderid"],
        ],
        ignore_index=True,
    )

    return df_order[~df_order["orderid"].isin(cancelled_ids)].copy().reset_index(drop=True)


def first_trade_price_after(df_trade: pd.DataFrame, time_value) -> Optional[float]:
    if df_trade is None or df_trade.empty:
        return None

    trade = df_trade.copy()
    trade["datetime"] = _as_datetime(trade["datetime"])
    valid = trade[(trade["price"] > 0) & (trade["datetime"].dt.time >= time_value)]
    if valid.empty:
        return None
    return float(valid.sort_values("datetime").iloc[0]["price"])


def first_actual_open_price(df_trade: pd.DataFrame) -> Optional[float]:
    return first_trade_price_after(df_trade, ACTUAL_OPEN_TIME)


def first_continuous_trade_price(df_trade: pd.DataFrame) -> Optional[float]:
    return first_trade_price_after(df_trade, CONTINUOUS_OPEN_TIME)


def build_candidate_table(
    df_buy: pd.DataFrame,
    df_sell: pd.DataFrame,
    min_price: float,
    max_price: float,
    prev_close: Optional[float] = None,
) -> pd.DataFrame:
    rows = []
    for price in generate_price_range(min_price, max_price):
        buy_vol = float(df_buy.loc[df_buy["price"] >= price, "size"].sum())
        sell_vol = float(df_sell.loc[df_sell["price"] <= price, "size"].sum())
        outside_buy_vol = float(df_buy.loc[df_buy["price"] < price, "size"].sum())
        outside_sell_vol = float(df_sell.loc[df_sell["price"] > price, "size"].sum())
        rows.append(
            {
                "price": float(price),
                "match_volume": min(buy_vol, sell_vol),
                "buy_vol": buy_vol,
                "sell_vol": sell_vol,
                "diff": abs(buy_vol - sell_vol),
                "outside_candidate_volume": outside_buy_vol + outside_sell_vol,
                "distance_to_prev_close": abs(float(price) - prev_close) if prev_close is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def filter_valid_candidates(candidates: pd.DataFrame, df_buy: pd.DataFrame, df_sell: pd.DataFrame) -> pd.DataFrame:
    valid_prices = []
    for _, row in candidates.iterrows():
        price = row["price"]
        higher_buy_vol = float(df_buy.loc[df_buy["price"] > price, "size"].sum())
        lower_sell_vol = float(df_sell.loc[df_sell["price"] < price, "size"].sum())

        if higher_buy_vol > row["sell_vol"]:
            continue
        if lower_sell_vol > row["buy_vol"]:
            continue
        valid_prices.append(price)

    return candidates[candidates["price"].isin(valid_prices)].copy().reset_index(drop=True)


def select_open_price(candidates: pd.DataFrame, stock_code: str, prev_close: Optional[float]) -> float:
    if candidates.empty:
        raise ValueError("没有可用候选价")

    max_volume = candidates["match_volume"].max()
    max_volume_candidates = candidates[candidates["match_volume"] == max_volume].copy()

    if stock_code.endswith(".SZ"):
        min_diff = max_volume_candidates["diff"].min()
        tied = max_volume_candidates[max_volume_candidates["diff"] == min_diff].copy()
        if len(tied) == 1 or prev_close is None:
            return float(tied.iloc[0]["price"])

        tied["distance_to_prev_close"] = (tied["price"] - float(prev_close)).abs()
        min_distance = tied["distance_to_prev_close"].min()
        nearest = tied[tied["distance_to_prev_close"] == min_distance]
        return float(nearest["price"].max())

    if stock_code.endswith(".SH"):
        if (candidates["diff"] == 0).all():
            return float(scaled_mean_round(candidates["price"], 2))

        edge_candidates = pd.concat([candidates.iloc[[0]], candidates.iloc[[-1]]])
        min_edge_diff = edge_candidates["diff"].min()
        tied_edges = edge_candidates[edge_candidates["diff"] == min_edge_diff]
        return float(scaled_mean_round(tied_edges["price"], 2))

    min_diff = max_volume_candidates["diff"].min()
    tied = max_volume_candidates[max_volume_candidates["diff"] == min_diff]
    return float(tied.iloc[0]["price"])


def _prepare_auction_orders(df_order: pd.DataFrame, df_trade: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    order = df_order.copy()
    order["datetime"] = _as_datetime(order["datetime"])
    order = order[order["datetime"].dt.time < AUCTION_ORDER_CUTOFF].copy()
    order = remove_cancelled_orders(order, df_trade)

    buy = order.loc[order["side"] == 1, ["orderid", "datetime", "price", "size"]].copy()
    sell = order.loc[order["side"] == -1, ["orderid", "datetime", "price", "size"]].copy()
    return buy.reset_index(drop=True), sell.reset_index(drop=True)


def _simulate_last_executable_boundary(df_buy: pd.DataFrame, df_sell: pd.DataFrame) -> tuple[Optional[float], Optional[float], float]:
    buy_book = df_buy.sort_values(["price", "datetime"], ascending=[False, True]).reset_index(drop=True).copy()
    sell_book = df_sell.sort_values(["price", "datetime"], ascending=[True, True]).reset_index(drop=True).copy()

    last_buy_price = None
    last_sell_price = None
    total_match_volume = 0.0

    while not buy_book.empty and not sell_book.empty:
        best_buy = buy_book.iloc[0]
        best_sell = sell_book.iloc[0]
        if best_buy["price"] < best_sell["price"]:
            break

        last_buy_price = float(best_buy["price"])
        last_sell_price = float(best_sell["price"])
        match_qty = min(float(best_buy["size"]), float(best_sell["size"]))
        total_match_volume += match_qty

        if float(best_buy["size"]) == match_qty:
            buy_book = buy_book.drop(buy_book.index[0]).reset_index(drop=True)
        else:
            buy_book.at[0, "size"] = float(buy_book.at[0, "size"]) - match_qty

        if float(best_sell["size"]) == match_qty:
            sell_book = sell_book.drop(sell_book.index[0]).reset_index(drop=True)
        else:
            sell_book.at[0, "size"] = float(sell_book.at[0, "size"]) - match_qty

    return last_buy_price, last_sell_price, total_match_volume


def calculate_open_price_for_frames(
    df_order: pd.DataFrame,
    df_trade: pd.DataFrame,
    prev_close: Optional[float],
    stock_code: str,
    trade_date: str,
) -> AuctionResult:
    actual_price = first_actual_open_price(df_trade)
    continuous_price = first_continuous_trade_price(df_trade)

    try:
        df_buy, df_sell = _prepare_auction_orders(df_order, df_trade)
        if df_buy.empty or df_sell.empty:
            return _failure_result(trade_date, stock_code, prev_close, actual_price, continuous_price, "买卖盘数据缺失")

        last_buy, last_sell, total_match_volume = _simulate_last_executable_boundary(df_buy, df_sell)
        if last_buy is None or last_sell is None or total_match_volume <= 0:
            return _failure_result(trade_date, stock_code, prev_close, actual_price, continuous_price, "集合竞价阶段无法形成可撮合订单")

        candidates = build_candidate_table(df_buy, df_sell, min(last_buy, last_sell), max(last_buy, last_sell), prev_close)
        valid_candidates = filter_valid_candidates(candidates, df_buy, df_sell)
        if valid_candidates.empty:
            return _failure_result(trade_date, stock_code, prev_close, actual_price, continuous_price, "没有满足交易所规则的候选价")

        selected = select_open_price(valid_candidates, stock_code, prev_close)
        error = abs(selected - actual_price) if actual_price is not None else None
        return AuctionResult(
            trade_date=trade_date,
            symbol=stock_code,
            prev_close=prev_close,
            calculated_open_price=selected,
            actual_open_price=actual_price,
            continuous_first_price=continuous_price,
            error=error,
            is_match=bool(error is not None and error < 0.005),
            status="success",
            message="撮合成功",
            candidate_count=len(valid_candidates),
            selected_rule=_selected_rule_name(stock_code),
        )
    except Exception as exc:
        return _failure_result(trade_date, stock_code, prev_close, actual_price, continuous_price, f"运行异常：{exc}")


def calculate_open_price_for_files(
    order_file: Path,
    trade_file: Path,
    prev_close: Optional[float],
    stock_code: str,
    trade_date: str,
) -> AuctionResult:
    df_order = pd.read_csv(order_file)
    df_trade = pd.read_csv(trade_file)
    return calculate_open_price_for_frames(df_order, df_trade, prev_close, stock_code, trade_date)


def _failure_result(
    trade_date: str,
    stock_code: str,
    prev_close: Optional[float],
    actual_price: Optional[float],
    continuous_price: Optional[float],
    message: str,
) -> AuctionResult:
    if continuous_price is not None:
        error = abs(continuous_price - actual_price) if actual_price is not None else None
        return AuctionResult(
            trade_date=trade_date,
            symbol=stock_code,
            prev_close=prev_close,
            calculated_open_price=continuous_price,
            actual_open_price=actual_price,
            continuous_first_price=continuous_price,
            error=error,
            is_match=bool(error is not None and error < 0.005),
            status="fallback_continuous",
            message=f"{message}，使用连续竞价第一笔成交价兜底",
        )

    return AuctionResult(
        trade_date=trade_date,
        symbol=stock_code,
        prev_close=prev_close,
        calculated_open_price=None,
        actual_open_price=actual_price,
        continuous_first_price=continuous_price,
        error=None,
        is_match=False,
        status="failed",
        message=message,
    )


def _selected_rule_name(stock_code: str) -> str:
    if stock_code.endswith(".SZ"):
        return "深市：最大成交量 -> diff 最小 -> 最接近前收盘价 -> 高价"
    if stock_code.endswith(".SH"):
        return "沪市：全部 diff=0 取全候选中间价；否则取候选首尾后按 diff 选边界/中间价"
    return "通用：最大成交量 -> diff 最小"
