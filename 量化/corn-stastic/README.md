# 双因子 + IC 自适应择时策略

## 策略概述

核心思想：**不用预测因子何时有效，而是让市场告诉我们因子现在有没有效。**

## 三层架构

### 第一层：两个子因子

**子因子 1 — CBOT 传导信号**

| CBOT 日涨幅 | 操作 |
|---|---|
| > +0.7% | 做多 DCE 玉米 (+1) |
| < -0.7% | 做空 DCE 玉米 (-1) |
| 其他 | 无信号 (0) |

**子因子 2 — 淀粉-玉米价差信号**

| 价差 Z-score | 操作 |
|---|---|
| < -1 | 做多 DCE 玉米 (+1) |
| > +1 | 做空 DCE 玉米 (-1) |
| 其他 | 无信号 (0) |

### 第二层：组合信号

```
组合信号 = 0.5 × CBOT信号 + 0.5 × 价差信号
```

- 组合 > +0.3 → 多头仓位 (+1)
- 组合 < -0.3 → 空头仓位 (-1)
- 其余 → 空仓 (0)

两个信号同向 → 强信号；方向相反 → 互相抵消

### 第三层：IC 自适应开关（核心增强）

每天收盘后：
1. 计算过去 60 天滚动 IC（CBOT收益 与 DCE收益 的 Pearson 相关性）
2. 计算过去 20 天 IC 均值
3. IC均值 > 0.03 → 执行组合信号
4. IC均值 ≤ 0.03 → 空仓观望

**今天发信号 → 明天开盘执行**

## 回测绩效

| 指标 | 数值 |
|---|---|
| 总收益 | +171.75% |
| 年化收益 | +9.70% |
| 夏普比率 | 1.31 |
| 最大回撤 | -14.30% |
| 手续费 | 万三（双边 0.03%） |
| 盈利年份 | 8 / 13 |

对比基准（买入持有）：总收益 -4.58%，最大回撤 -56%

## 文件结构

```
corn-stastic/
├── 玉米期货数据.xlsx          # 原始数据
├── data_loader.py             # 数据加载与预处理
├── dual_factor_ic_strategy.py # 策略核心逻辑
├── backtest_engine.py         # 回测引擎 + 图表
├── parameter_analysis.py      # 参数敏感性分析
├── run_strategy.py            # 主入口脚本
├── requirements.txt
└── README.md
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

```bash
# 完整回测 + 生成所有图表
python run_strategy.py

# 仅查看最新信号
python run_strategy.py --mode latest

# 参数敏感性分析
python run_strategy.py --mode sensitivity

# 指定数据文件路径
python run_strategy.py --excel /path/to/数据.xlsx

# 指定输出目录
python run_strategy.py --out-dir ./my_results
```

## 输出文件（results/）

| 文件 | 说明 |
|---|---|
| `equity_curve.png` | 净值曲线 vs 买入持有基准 + 回撤图 |
| `ic_analysis.png` | 滚动 IC、IC均值、IC激活状态 |
| `annual_returns.png` | 年度收益柱状图 |
| `factor_signals.png` | 子因子信号时序图 |
| `daily_results.csv` | 每日完整结果数据 |
| `grid_search_results.csv` | 全参数网格搜索排名 |
| `heatmap_sharpe.png` | 夏普热力图 |
| `sens_*.csv/png` | 各参数敏感性数据与图表 |

## 核心参数（可调整）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `cbot_threshold` | 0.007 | CBOT 涨跌幅阈值 0.7% |
| `spread_zscore_threshold` | 1.0 | 价差 Z-score 阈值 ±1 |
| `combined_threshold` | 0.3 | 组合信号阈值 ±0.3 |
| `ic_window` | 60 | 滚动 IC 窗口（天） |
| `ic_mean_window` | 20 | IC 均值窗口（天） |
| `ic_threshold` | 0.03 | IC 激活阈值 |

## 后续可探索方向

1. **参数敏感性**：对 0.7% CBOT 阈值、1.0 价差阈值、0.03 IC 阈值进行微调优化
2. **分批建仓**：一次开满 vs 分 2-3 次，可以降低滑点
3. **实盘模拟**：用最近 6 个月做模拟盘验证
