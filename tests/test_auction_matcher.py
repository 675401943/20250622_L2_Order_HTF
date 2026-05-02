import pandas as pd

from src.auction_matcher import (
    build_candidate_table,
    calculate_open_price_for_frames,
    first_actual_open_price,
    remove_cancelled_orders,
    select_open_price,
)


def test_candidate_table_uses_cumulative_quantity_difference():
    df_buy = pd.DataFrame(
        [
            {"orderid": 1, "datetime": "2000-01-01 09:15:01", "price": 10.02, "size": 100},
            {"orderid": 2, "datetime": "2000-01-01 09:15:02", "price": 10.01, "size": 100},
        ]
    )
    df_sell = pd.DataFrame(
        [
            {"orderid": 3, "datetime": "2000-01-01 09:15:03", "price": 10.00, "size": 100},
            {"orderid": 4, "datetime": "2000-01-01 09:15:04", "price": 10.01, "size": 50},
        ]
    )

    table = build_candidate_table(df_buy, df_sell, 10.00, 10.02, prev_close=10.01)
    row = table.loc[table["price"] == 10.01].iloc[0]

    assert row["buy_vol"] == 200
    assert row["sell_vol"] == 150
    assert row["match_volume"] == 150
    assert row["diff"] == 50


def test_sz_tie_breaks_by_prev_close_then_higher_price():
    candidates = pd.DataFrame(
        [
            {"price": 10.00, "match_volume": 1000, "buy_vol": 1000, "sell_vol": 1000, "diff": 0},
            {"price": 10.02, "match_volume": 1000, "buy_vol": 1000, "sell_vol": 1000, "diff": 0},
        ]
    )

    selected = select_open_price(candidates, "ANON001.SZ", prev_close=10.01)

    assert selected == 10.02


def test_sh_all_zero_diff_uses_mid_price_with_rounding():
    candidates = pd.DataFrame(
        [
            {"price": 16.49, "match_volume": 3200, "buy_vol": 3200, "sell_vol": 3200, "diff": 0},
            {"price": 16.50, "match_volume": 3200, "buy_vol": 3200, "sell_vol": 3200, "diff": 0},
        ]
    )

    selected = select_open_price(candidates, "ANON002.SH", prev_close=16.43)

    assert selected == 16.50


def test_sh_non_zero_diff_uses_edge_candidates_like_notebook_logic():
    candidates = pd.DataFrame(
        [
            {"price": 10.00, "match_volume": 1000, "buy_vol": 1000, "sell_vol": 1000, "diff": 0},
            {"price": 10.01, "match_volume": 1000, "buy_vol": 1000, "sell_vol": 1000, "diff": 0},
            {"price": 10.02, "match_volume": 1000, "buy_vol": 1000, "sell_vol": 1000, "diff": 0},
            {"price": 10.03, "match_volume": 1000, "buy_vol": 1100, "sell_vol": 1000, "diff": 100},
        ]
    )

    selected = select_open_price(candidates, "ANON003.SH", prev_close=9.99)

    assert selected == 10.00


def test_remove_cancelled_orders_before_auction_cutoff():
    df_order = pd.DataFrame(
        [
            {"orderid": 1, "datetime": "2000-01-01 09:15:01", "price": 10.01, "size": 100},
            {"orderid": 2, "datetime": "2000-01-01 09:15:02", "price": 10.02, "size": 100},
        ]
    )
    df_trade = pd.DataFrame(
        [
            {
                "datetime": "2000-01-01 09:20:00",
                "price": 0,
                "size": 0,
                "bidorderid": 1,
                "askorderid": 0,
            },
            {
                "datetime": "2000-01-01 09:27:00",
                "price": 0,
                "size": 0,
                "bidorderid": 2,
                "askorderid": 0,
            },
        ]
    )

    filtered = remove_cancelled_orders(df_order, df_trade)

    assert filtered["orderid"].tolist() == [2]


def test_fallback_to_first_continuous_trade_when_auction_cannot_match():
    df_order = pd.DataFrame(
        [
            {"orderid": 1, "datetime": "2000-01-01 09:15:01", "sym": "ANON001.SZ", "price": 9.90, "size": 100, "side": 1},
            {"orderid": 2, "datetime": "2000-01-01 09:15:02", "sym": "ANON001.SZ", "price": 10.10, "size": 100, "side": -1},
        ]
    )
    df_trade = pd.DataFrame(
        [
            {
                "datetime": "2000-01-01 09:30:01",
                "sym": "ANON001.SZ",
                "price": 10.00,
                "size": 100,
                "bidorderid": 10,
                "askorderid": 11,
            }
        ]
    )

    result = calculate_open_price_for_frames(df_order, df_trade, 9.95, "ANON001.SZ", "2000-01-01")

    assert result.status == "fallback_continuous"
    assert result.calculated_open_price == 10.00
    assert result.actual_open_price == 10.00
    assert result.is_match is True


def test_first_actual_open_price_handles_mixed_fractional_seconds():
    df_trade = pd.DataFrame(
        [
            {
                "datetime": "2000-01-01 09:30:00.330",
                "sym": "ANON001.SZ",
                "price": 0.0,
                "size": 100,
                "bidorderid": 1,
                "askorderid": 0,
            },
            {
                "datetime": "2000-01-01 09:30:01",
                "sym": "ANON001.SZ",
                "price": 5.86,
                "size": 200,
                "bidorderid": 1,
                "askorderid": 2,
            },
            {
                "datetime": "2000-01-01 09:30:01.200",
                "sym": "ANON001.SZ",
                "price": 5.85,
                "size": 1000,
                "bidorderid": 3,
                "askorderid": 2,
            },
        ]
    )

    assert first_actual_open_price(df_trade) == 5.86
