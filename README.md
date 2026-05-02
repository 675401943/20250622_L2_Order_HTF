# L2 逐笔委托开盘价撮合

这个项目整理了一套从 Level-2 逐笔委托、逐笔成交数据中复现股票开盘集合竞价成交价的流程。项目目标不是训练机器学习模型，而是把交易所集合竞价规则转成可复现、可验证的代码。

## 项目亮点

- 将原始 notebook 重构为可运行的 Python 项目。
- 将数据整理为 `train`、`validation`、`test` 三份，便于说明开发、验证和样本外测试过程。
- 对公开仓库中的样本文件、股票代码和交易日期做脱敏处理，只保留撮合逻辑需要的市场后缀和价格数量信息。
- 用单元测试锁定关键规则，尤其是“未成交量最小”的正确理解。
- 用中文文档解释题目、规则、数据和结果，方便没有交易所撮合背景的人阅读。
- 最终全量运行结果：`train`、`validation`、`test` 均命中实际开盘价。

## 目录结构

```text
data/
  train/          # 14 个开发样本
  validation/     # 6 个验证样本
  test/           # 279 个样本外测试样本
notes/
  opening_auction_notes.md
src/
  auction_matcher.py
  run_pipeline.py
scripts/
  prepare_dataset.py
tests/
  test_auction_matcher.py
  test_prepare_dataset.py
results/
  summary.md
```

## 数据脱敏说明

公开仓库不保留原始文件名、真实股票代码或真实交易日期，统一改成下面的匿名格式：

- 文件名：`TRN001_order.csv`、`VAL001_trade.csv`、`TST001_order.csv`。
- 样本编号：`TRN` 表示训练集，`VAL` 表示验证集，`TST` 表示测试集。
- 股票代码：`ANON001.SZ`、`ANON002.SH`，仅保留交易所后缀用于区分深市和沪市规则。
- 交易日期：训练集从 `2000-01-01` 起编号，验证集从 `2001-01-01` 起编号，测试集从 `2002-01-01` 起编号。

脱敏不会改变同一样本内的时间顺序、价格、数量、买卖方向和成交关系，因此不影响集合竞价撮合逻辑。

## 如何运行

安装依赖：

```bash
pip install pandas numpy pytest openpyxl
```

运行单元测试：

```bash
python -m pytest tests -q
```

重新生成数据集：

```bash
python scripts/prepare_dataset.py --source-root ..
```

运行全部撮合流程：

```bash
python src/run_pipeline.py --split all
```

也可以只跑某一份数据：

```bash
python src/run_pipeline.py --split train
python src/run_pipeline.py --split validation
python src/run_pipeline.py --split test
```

## 输出结果

运行后会生成：

- `results/train_results.csv`
- `results/validation_results.csv`
- `results/test_results.csv`
- `results/summary.md`

`summary.md` 是最适合快速展示的结果页，包含每份数据的样本数、成功撮合数、命中数、准确率和规则说明。

## 核心规则

集合竞价开盘价不是简单取某一笔成交价，而是在候选价格中按规则筛选：

1. 可实现最大成交量；
2. 高于该价格的买入申报、低于该价格的卖出申报全部成交；
3. 如果仍有多个价格，比较买卖累计申报数量差；
4. 深市和沪市对并列价格有不同处理。

本项目中的关键修正是：

```text
“未成交量最小” = 买卖累计申报数量差最小 = abs(buy_vol - sell_vol)
```

它不是简单把所有未参与成交的低价买单、高价卖单相加。

## 文档入口

- [题目说明](problem_statement.md)
- [集合竞价笔记](notes/opening_auction_notes.md)
- [运行结果汇总](results/summary.md)
