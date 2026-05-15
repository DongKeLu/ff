"""
玉米期货价格预测 — 多时间尺度增强版
==============================================
新增功能：
  1. 多时间尺度预测：次日(1天)、1周、2周、4周、8周（≈半月）
  2. 趋势强度评分（0-100），区分 强势上涨/上涨/震荡/下跌/强势下跌
  3. 方向概率引擎：Monte Carlo 模拟，给出各方向概率分布
  4. 分层 Walk-Forward 验证：每个时间尺度独立评估泛化能力
  5. 每日/每周/半月 三层预测报告 + 置信区间
  6. Walk-Forward 回测胜率统计（分方向）

运行方式: python corn_multi_horizon.py
依赖: pandas numpy matplotlib scikit-learn xgboost scipy
"""

import pandas as pd
import numpy as np
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = [
    'Heiti TC', 'STHeiti', 'SimHei', 'Songti SC',
    'WenQuanYi Micro Hei', 'Arial Unicode MS', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

os.chdir('/root/ff/量化/其他策略')

# ============================================================
#  PART 1：数据加载 & 预处理
# ============================================================
file_path = '玉米加权 -512.xlsx'
df_raw = pd.read_excel(file_path, sheet_name='玉米加权')

col_map = {
    '日期': 'date', '开盘价': 'open', '最高价': 'high', '最低价': 'low',
    '收盘价': 'close', '成交量': 'volume', '持仓量': 'oi', '结算价': 'settle',
    '华北小麦替代利润': 'wheat_profit'
}
df = df_raw.rename(columns=col_map).copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

num_cols = ['open', 'high', 'low', 'close', 'volume', 'oi', 'settle', 'wheat_profit']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ============================================================
#  PART 2：构造 日线、周线 两套特征集
# ============================================================
# ---- 日线（用于次日预测）----
df_daily = df[['date', 'open', 'high', 'low', 'close', 'volume', 'oi', 'settle']].copy()
if 'wheat_profit' in df.columns:
    df_daily = df_daily.merge(df[['date', 'wheat_profit']], on='date', how='left')
df_daily = df_daily.dropna(subset=['close']).reset_index(drop=True)

# 日线基础指标
df_daily['ret_d1'] = df_daily['close'].pct_change(1)
df_daily['ret_d5'] = df_daily['close'].pct_change(5)
df_daily['ret_d10'] = df_daily['close'].pct_change(10)
df_daily['ret_d20'] = df_daily['close'].pct_change(20)
df_daily['vol_d5'] = df_daily['ret_d1'].rolling(5).std()
df_daily['vol_d10'] = df_daily['ret_d1'].rolling(10).std()
df_daily['vol_d20'] = df_daily['ret_d1'].rolling(20).std()

df_daily['ma5_d'] = df_daily['close'].rolling(5).mean()
df_daily['ma10_d'] = df_daily['close'].rolling(10).mean()
df_daily['ma20_d'] = df_daily['close'].rolling(20).mean()
df_daily['ma60_d'] = df_daily['close'].rolling(60).mean()

df_daily['bias_ma5'] = (df_daily['close'] - df_daily['ma5_d']) / df_daily['ma5_d']
df_daily['bias_ma20'] = (df_daily['close'] - df_daily['ma20_d']) / df_daily['ma20_d']
df_daily['bias_ma60'] = (df_daily['close'] - df_daily['ma60_d']) / df_daily['ma60_d']

df_daily['upper_band'] = df_daily['ma20_d'] + 2 * df_daily['close'].rolling(20).std()
df_daily['lower_band'] = df_daily['ma20_d'] - 2 * df_daily['close'].rolling(20).std()
df_daily['band_width'] = (df_daily['upper_band'] - df_daily['lower_band']) / df_daily['ma20_d']
df_daily['band_pos'] = (df_daily['close'] - df_daily['lower_band']) / (
    df_daily['upper_band'] - df_daily['lower_band'])

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    return rsi

df_daily['rsi_14'] = compute_rsi(df_daily['close'], 14)

vol_ma5_d = df_daily['volume'].rolling(5).mean()
df_daily['vol_ratio_d'] = df_daily['volume'] / vol_ma5_d
df_daily['oi_chg_d1'] = df_daily['oi'].pct_change(1)
df_daily['oi_chg_d5'] = df_daily['oi'].pct_change(5)
df_daily['range_pct_d'] = (df_daily['high'] - df_daily['low']) / df_daily['close']

ema12 = df_daily['close'].ewm(span=12, adjust=False).mean()
ema26 = df_daily['close'].ewm(span=26, adjust=False).mean()
df_daily['macd'] = ema12 - ema26
df_daily['macd_signal'] = df_daily['macd'].ewm(span=9, adjust=False).mean()
df_daily['macd_hist'] = df_daily['macd'] - df_daily['macd_signal']

if 'settle' in df_daily.columns:
    df_daily['settle_spread'] = df_daily['settle'] - df_daily['close']

has_wheat_d = 'wheat_profit' in df_daily.columns and df_daily['wheat_profit'].notna().sum() > 20
if has_wheat_d:
    df_daily['wheat_ma4'] = df_daily['wheat_profit'].rolling(4).mean()

daily_features = [
    'ret_d1', 'ret_d5', 'ret_d10', 'ret_d20',
    'vol_d5', 'vol_d10', 'vol_d20',
    'bias_ma5', 'bias_ma20', 'bias_ma60',
    'band_width', 'band_pos',
    'rsi_14',
    'vol_ratio_d', 'oi_chg_d1', 'oi_chg_d5', 'range_pct_d',
    'macd', 'macd_signal', 'macd_hist',
]
if has_wheat_d:
    daily_features += ['wheat_profit', 'wheat_ma4']
if 'settle_spread' in df_daily.columns:
    daily_features += ['settle_spread']

# ---- 周线（用于1周/2周/4周/8周预测）----
agg_w = {
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'oi': 'last', 'settle': 'last',
}
if 'wheat_profit' in df.columns:
    agg_w['wheat_profit'] = 'last'

df_w = df.set_index('date').resample('W-MON').agg(agg_w).dropna(subset=['close']).reset_index()
df_w = df_w.sort_values('date').reset_index(drop=True)
df_w['year'] = df_w['date'].dt.isocalendar().year.astype(int)
df_w['week'] = df_w['date'].dt.isocalendar().week.astype(int)
df_w['year_week'] = df_w['year'].astype(str) + '_' + df_w['week'].astype(str).str.zfill(2)

# 周线基础指标
df_w['ret_1w'] = df_w['close'].pct_change(1)
df_w['ret_2w'] = df_w['close'].pct_change(2)
df_w['ret_4w'] = df_w['close'].pct_change(4)
df_w['ret_8w'] = df_w['close'].pct_change(8)
df_w['mom_diff'] = df_w['ret_1w'] - df_w['ret_4w']
df_w['mom_accel'] = df_w['ret_1w'] - 2 * df_w['ret_4w'] + df_w['ret_8w']

df_w['ma5'] = df_w['close'].rolling(5).mean()
df_w['ma10'] = df_w['close'].rolling(10).mean()
df_w['ma20'] = df_w['close'].rolling(20).mean()
df_w['bias_5'] = (df_w['close'] - df_w['ma5']) / df_w['ma5']
df_w['bias_10'] = (df_w['close'] - df_w['ma10']) / df_w['ma10']
df_w['bias_20'] = (df_w['close'] - df_w['ma20']) / df_w['ma20']

df_w['vol_4w'] = df_w['ret_1w'].rolling(4).std()
df_w['vol_8w'] = df_w['ret_1w'].rolling(8).std()
df_w['vol_16w'] = df_w['ret_1w'].rolling(16).std()
df_w['range_pct'] = (df_w['high'] - df_w['low']) / df_w['close']

vol_ma5 = df_w['volume'].rolling(5).mean()
df_w['vol_ratio'] = df_w['volume'] / vol_ma5
df_w['oi_chg'] = df_w['oi'].pct_change(1)
df_w['oi_chg_4w'] = df_w['oi'].pct_change(4)
df_w['oi_chg_8w'] = df_w['oi'].pct_change(8)

ema12_w = df_w['close'].ewm(span=12, adjust=False).mean()
ema26_w = df_w['close'].ewm(span=26, adjust=False).mean()
df_w['macd'] = ema12_w - ema26_w
df_w['macd_signal'] = df_w['macd'].ewm(span=9, adjust=False).mean()
df_w['macd_hist'] = df_w['macd'] - df_w['macd_signal']

has_wheat = 'wheat_profit' in df_w.columns and df_w['wheat_profit'].notna().sum() > 20
if has_wheat:
    df_w['wheat_profit_ma4'] = df_w['wheat_profit'].rolling(4).mean()
    df_w['wheat_trend'] = df_w['wheat_profit'].pct_change(4)
    print("华北小麦替代利润已加入")

if 'settle' in df_w.columns:
    df_w['settle_spread'] = df_w['settle'] - df_w['close']

weekly_features = [
    'ret_1w', 'ret_2w', 'ret_4w', 'ret_8w',
    'mom_diff', 'mom_accel',
    'bias_5', 'bias_10', 'bias_20',
    'vol_4w', 'vol_8w', 'vol_16w',
    'range_pct',
    'vol_ratio', 'oi_chg', 'oi_chg_4w', 'oi_chg_8w',
    'macd', 'macd_signal', 'macd_hist',
]
if has_wheat:
    weekly_features += ['wheat_profit', 'wheat_profit_ma4', 'wheat_trend']
if 'settle_spread' in df_w.columns:
    weekly_features += ['settle_spread']

print(f"日线特征数: {len(daily_features)}, 周线特征数: {len(weekly_features)}")
print(f"日线数据: {df_daily['date'].min().strftime('%Y-%m-%d')} → {df_daily['date'].max().strftime('%Y-%m-%d')}  共 {len(df_daily)} 天")
print(f"周线数据: {df_w['date'].min().strftime('%Y-%m-%d')} → {df_w['date'].max().strftime('%Y-%m-%d')}  共 {len(df_w)} 周")
print(f"  ⚠️ 周线使用 W-MON 聚合，5-12(周二) 被归入 {df_w['date'].max().strftime('%Y-%m-%d')} 周，预测基准仍为5-12收盘")


# ============================================================
#  PART 3：构建多目标标签（日/周/半月）
# ============================================================
# 日线目标：次日收盘价
df_daily['target_next1d'] = df_daily['close'].shift(-1)
df_daily['target_ret_next1d'] = df_daily['close'].pct_change(1).shift(-1)

# 周线目标：1周后、2周后、4周后、8周后的收盘价
df_w['target_1w'] = df_w['close'].shift(-1)
df_w['target_2w'] = df_w['close'].shift(-2)
df_w['target_4w'] = df_w['close'].shift(-4)
df_w['target_8w'] = df_w['close'].shift(-8)

df_w['target_ret_1w'] = df_w['close'].pct_change(1).shift(-1)
df_w['target_ret_2w'] = df_w['close'].pct_change(2).shift(-2)
df_w['target_ret_4w'] = df_w['close'].pct_change(4).shift(-4)
df_w['target_ret_8w'] = df_w['close'].pct_change(8).shift(-8)


# ============================================================
#  PART 4：准备日线训练数据
# ============================================================
target_daily = 'target_next1d'
df_d_v2 = df_daily.dropna(subset=daily_features + [target_daily]).reset_index(drop=True)
print(f"\n日线有效样本: {len(df_d_v2)} 天")

X_d = df_d_v2[daily_features].values
y_d = df_d_v2[target_daily].values
dates_d = df_d_v2['date'].values

scaler_d = StandardScaler()
X_d_scaled = scaler_d.fit_transform(X_d)

# ============================================================
#  PART 5：准备周线训练数据（每个时间尺度独立）
# ============================================================
horizons = {
    '1w': {'target': 'target_1w', 'ret_target': 'target_ret_1w', 'n_step': 1},
    '2w': {'target': 'target_2w', 'ret_target': 'target_ret_2w', 'n_step': 2},
    '4w': {'target': 'target_4w', 'ret_target': 'target_ret_4w', 'n_step': 4},
    '8w': {'target': 'target_8w', 'ret_target': 'target_ret_8w', 'n_step': 8},
}

# 过滤掉周线中包含无穷大/NaN的行，保持索引对齐
df_w_clean = df_w.replace([np.inf, -np.inf], np.nan)
mask_w = ~df_w_clean[weekly_features].isna().any(axis=1)
df_w_valid = df_w[mask_w].reset_index(drop=True)

scaler_w = StandardScaler()
X_w_all = scaler_w.fit_transform(df_w_valid[weekly_features].values)
y_w = {k: df_w_valid[v['target']].values for k, v in horizons.items()}
y_w_ret = {k: df_w_valid[v['ret_target']].values for k, v in horizons.items()}
dates_w = df_w_valid['date'].values

print(f"周线有效样本: {len(df_w_valid)} 周（已剔除{np.sum(~mask_w)}个异常行）")
df_w = df_w_valid  # use cleaned df for all subsequent operations


# ============================================================
#  PART 6：模型工厂
# ============================================================
def build_elasticnet(Xs, y_):
    m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                     n_alphas=200, cv=5, max_iter=30000, random_state=42, n_jobs=-1)
    m.fit(Xs, y_)
    model = ElasticNet(alpha=m.alpha_, l1_ratio=m.l1_ratio_, max_iter=30000, random_state=42)
    model.fit(Xs, y_)
    return model

def build_ridge(Xs, y_):
    m = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
    m.fit(Xs, y_)
    return m

def build_rf(Xs, y_):
    m = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=3,
        max_features=0.7, random_state=42, n_jobs=-1
    )
    m.fit(Xs, y_)
    return m

def build_xgb(Xs, y_):
    from xgboost import XGBRegressor
    m = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    m.fit(Xs, y_)
    return m

factories = {
    'ElasticNet': build_elasticnet,
    'Ridge': build_ridge,
    'RandomForest': build_rf,
    'XGBoost': build_xgb,
}

colors_model = {
    'ElasticNet': '#E91E63', 'Ridge': '#FF9800',
    'RandomForest': '#2196F3', 'XGBoost': '#4CAF50'
}


# ============================================================
#  PART 7：训练日线模型（预测次日价格）
# ============================================================
print("\n" + "=" * 70)
print("  训练日线模型（预测次日收盘价）")
print("=" * 70)

daily_results = {}
daily_models = {}
for name, factory in factories.items():
    try:
        model = factory(X_d_scaled, y_d)
        yp = model.predict(X_d_scaled)
        r2 = r2_score(y_d, yp)
        mae = mean_absolute_error(y_d, yp)
        mape = mean_absolute_percentage_error(y_d, yp) * 100
        resid_std = np.std(y_d - yp)
        print(f"  {name:<15} R²={r2:.4f}  MAE={mae:.2f}  MAPE={mape:.2f}%  σ=±{resid_std:.2f}")
        daily_results[name] = {'r2': r2, 'mae': mae, 'mape': mape, 'resid_std': resid_std, 'y_pred': yp}
        daily_models[name] = model
    except Exception as e:
        print(f"  ⚠️ {name} 失败: {e}")

best_daily_name = max(daily_results, key=lambda k: daily_results[k]['r2'])
best_daily_model = daily_models[best_daily_name]
print(f"\n  ✅ 最佳日线模型: {best_daily_name} (R²={daily_results[best_daily_name]['r2']:.4f})")


# ============================================================
#  PART 8：训练周线模型（每个时间尺度独立）
# ============================================================
print("\n" + "=" * 70)
print("  训练周线模型（1周/2周/4周/8周 四个时间尺度）")
print("=" * 70)

weekly_results = {}
weekly_models = {}

for h_name, h_cfg in horizons.items():
    target = h_cfg['target']
    valid_mask = ~np.isnan(df_w[target].values)
    X_h = X_w_all[valid_mask]
    y_h = df_w.loc[valid_mask, target].values
    dates_h = dates_w[valid_mask]

    n_valid = valid_mask.sum()
    print(f"\n  [{h_name}] 有效样本: {n_valid} 周")

    h_results = {}
    h_models = {}
    for name, factory in factories.items():
        try:
            model = factory(X_h, y_h)
            yp = model.predict(X_h)
            r2 = r2_score(y_h, yp)
            mae = mean_absolute_error(y_h, yp)
            mape = mean_absolute_percentage_error(y_h, yp) * 100
            resid_std = np.std(y_h - yp)
            h_results[name] = {'r2': r2, 'mae': mae, 'mape': mape, 'resid_std': resid_std, 'y_pred': yp, 'n': n_valid}
            h_models[name] = model
        except Exception as e:
            print(f"    ⚠️ {name} 失败: {e}")

    best_h = max(h_results, key=lambda k: h_results[k]['r2'])
    print(f"  ✅ 最佳 [{h_name}] 模型: {best_h} (R²={h_results[best_h]['r2']:.4f})")

    weekly_results[h_name] = {'results': h_results, 'best': best_h, 'best_result': h_results[best_h]}
    weekly_models[h_name] = {'models': h_models, 'best': h_models[best_h]}


# ============================================================
#  PART 9：趋势强度评分引擎
# ============================================================
def compute_trend_strength(df_ref, n_weeks=8):
    """
    综合多个指标计算趋势强度（0-100）
    0-20: 强势下跌, 20-40: 下跌, 40-60: 震荡, 60-80: 上涨, 80-100: 强势上涨
    """
    scores = []

    # 1. 价格动量 (权重 25%)
    ret = df_ref['close'].pct_change(4).iloc[-1] if len(df_ref) >= 5 else 0
    momentum_score = np.clip(50 + ret * 500, 0, 100)
    scores.append(('动量', momentum_score, 0.25))

    # 2. 均线多头排列 (权重 20%)
    ma5 = df_ref['close'].rolling(5).mean().iloc[-1]
    ma10 = df_ref['close'].rolling(10).mean().iloc[-1]
    ma20 = df_ref['close'].rolling(20).mean().iloc[-1]
    close = df_ref['close'].iloc[-1]
    if len(df_ref) < 20:
        ma_score = 50
    elif ma5 > ma10 > ma20 and close > ma5:
        ma_score = 85
    elif ma5 > ma10 and close > ma10:
        ma_score = 70
    elif ma5 < ma10 < ma20 and close < ma5:
        ma_score = 15
    elif ma5 < ma10 and close < ma10:
        ma_score = 30
    else:
        ma_score = 50
    scores.append(('均线', ma_score, 0.20))

    # 3. MACD (权重 15%)
    ema12 = df_ref['close'].ewm(span=12).mean().iloc[-1]
    ema26 = df_ref['close'].ewm(span=26).mean().mean()
    macd = df_ref['close'].ewm(span=12).mean().iloc[-1] - df_ref['close'].ewm(span=26).mean().iloc[-1]
    macd_signal = df_ref['close'].ewm(span=9).mean().iloc[-1]
    if len(df_ref) < 26:
        macd_score = 50
    elif macd > macd_signal and macd > 0:
        macd_score = 80
    elif macd > macd_signal:
        macd_score = 65
    elif macd < macd_signal and macd < 0:
        macd_score = 20
    else:
        macd_score = 35
    scores.append(('MACD', macd_score, 0.15))

    # 4. RSI (权重 15%)
    if len(df_ref) >= 15:
        rsi = compute_rsi(df_ref['close'], 14).iloc[-1]
    else:
        rsi = 50
    rsi_score = float(np.clip(rsi, 0, 100))
    scores.append(('RSI', rsi_score, 0.15))

    # 5. 持仓量变化趋势 (权重 15%)
    if 'oi' in df_ref.columns and len(df_ref) >= 5:
        oi_trend = df_ref['oi'].pct_change(4).iloc[-1]
        if oi_trend > 0.1:
            oi_score = 70
        elif oi_trend > 0.05:
            oi_score = 60
        elif oi_trend < -0.1:
            oi_score = 30
        elif oi_trend < -0.05:
            oi_score = 40
        else:
            oi_score = 50
    else:
        oi_score = 50
    scores.append(('持仓', oi_score, 0.15))

    # 6. 波动率偏离 (权重 10%)
    vol_current = df_ref['close'].pct_change().iloc[-4:].std()
    vol_hist = df_ref['close'].pct_change().rolling(20).std().iloc[-1]
    if vol_hist > 0 and len(df_ref) >= 20:
        vol_ratio = vol_current / vol_hist
        vol_score = np.clip(50 + (1 - vol_ratio) * 30, 20, 80)
    else:
        vol_score = 50
    scores.append(('波动', vol_score, 0.10))

    # 加权得分
    total_score = sum(s[1] * s[2] for s in scores)
    return total_score, scores

def trend_label(score):
    if score >= 80: return '强势上涨', '#D32F2F'
    elif score >= 60: return '上涨', '#388E3C'
    elif score >= 40: return '震荡', '#F57C00'
    elif score >= 20: return '下跌', '#1976D2'
    else: return '强势下跌', '#7B1FA2'


# ============================================================
#  PART 10：Monte Carlo 方向概率引擎
# ============================================================
def directional_probability_monte_carlo(model, scaler, features, n_simulations=3000, seed=42):
    """
    对未来某时间尺度做 Monte Carlo 模拟：
    1. 从模型残差分布中采样噪声
    2. 生成概率分布
    3. 计算上涨/震荡/下跌概率
    """
    np.random.seed(seed)
    feat_vec = np.array([features[c] for c in features]).reshape(1, -1)
    feat_scaled = scaler.transform(feat_vec)
    base_pred = model.predict(feat_scaled)[0]

    # 残差分布估计（简化：使用固定波动率模型）
    # 假设预测误差服从正态分布，sigma 由近期滚动波动率估计
    recent_vol = features.get('vol_4w', 0.01) if 'vol_4w' in features else 0.01
    vol_daily = features.get('vol_d5', 0.005) if 'vol_d5' in features else 0.005

    if 'vol_4w' in features:
        sigma = max(recent_vol, 0.005) * 2
    elif 'vol_d5' in features:
        sigma = max(vol_daily * np.sqrt(5), 0.003)
    else:
        sigma = 0.01

    simulations = np.random.normal(base_pred, sigma * np.abs(base_pred) if base_pred != 0 else sigma, n_simulations)

    # 计算方向概率
    current_price = features.get('current_price', base_pred)

    p_up = np.mean(simulations > current_price * 1.005)    # 上涨 > 0.5%
    p_down = np.mean(simulations < current_price * 0.995)   # 下跌 < -0.5%
    p_flat = 1 - p_up - p_down

    return {
        'base_pred': base_pred,
        'simulations': simulations,
        'prob_up': p_up,
        'prob_flat': p_flat,
        'prob_down': p_down,
        'p5': np.percentile(simulations, 5),
        'p25': np.percentile(simulations, 25),
        'p50': np.percentile(simulations, 50),
        'p75': np.percentile(simulations, 75),
        'p95': np.percentile(simulations, 95),
        'mean_sim': np.mean(simulations),
    }


# ============================================================
#  PART 11：分层 Walk-Forward 验证
# ============================================================
print("\n" + "=" * 70)
print("  Walk-Forward 分层验证（每个时间尺度独立评估）")
print("=" * 70)

train_window = 80
step_size = 5

def walk_forward_evaluate(X_all, y_all, dates_all, n_steps, window=80, step=5,
                           factories_ref=None):
    """对某个 horizon 的目标做 Walk-Forward，返回各模型的表现"""
    valid_mask = ~np.isnan(y_all)
    X_v = X_all[valid_mask]
    y_v = y_all[valid_mask]
    d_v = dates_all[valid_mask]
    n_total = len(y_v)

    wf_results = {}
    for name in factories_ref:
        wf_results[name] = {'preds': [], 'actuals': [], 'dates': []}

    for train_end in range(window, n_total, step):
        train_start = train_end - window
        test_idx = train_end

        X_train = X_v[train_start:train_end]
        y_train = y_v[train_start:train_end]
        X_test = X_v[test_idx:test_idx+1]
        y_test = y_v[test_idx]

        for name in factories_ref:
            if name == 'ElasticNet':
                m = ElasticNet(alpha=16.4, l1_ratio=1.0, max_iter=20000, random_state=42)
            elif name == 'Ridge':
                m = Ridge(alpha=1.0)
            elif name == 'RandomForest':
                m = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5,
                                          max_features=0.7, random_state=42, n_jobs=-1)
            elif name == 'XGBoost':
                from xgboost import XGBRegressor
                m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.7,
                                  reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, n_jobs=-1, verbosity=0)

            m.fit(X_train, y_train)
            pred = m.predict(X_test)[0]
            wf_results[name]['preds'].append(pred)
            wf_results[name]['actuals'].append(y_test)
            wf_results[name]['dates'].append(dts_all[test_idx])

    return wf_results


def compute_wf_summary(wf_results, models_results):
    summary = {}
    for name in wf_results:
        preds = np.array(wf_results[name]['preds'])
        actuals = np.array(wf_results[name]['actuals'])
        if len(preds) == 0:
            continue
        wf_r2 = r2_score(actuals, preds)
        wf_mae = mean_absolute_error(actuals, preds)
        wf_mape = mean_absolute_percentage_error(actuals, preds) * 100
        train_r2 = models_results.get(name, {}).get('r2', np.nan)
        overfit = train_r2 - wf_r2 if not np.isnan(train_r2) else np.nan

        # 方向胜率：预测涨跌方向是否与实际涨跌方向一致
        # 对齐：preds[:-1] 和 actuals[:-1] 构成 "上一时间点的预测 vs 上一时间点的实际"
        #       actuals[1:] - actuals[:-1] 是实际涨跌
        #       preds[1:] - preds[:-1] 是预测涨跌（但预测在 walk-forward 时对应 actuals[1:]）
        # 简化：比较 predicted_direction = sign(preds[1:] - actuals[:-1]) vs actual_direction = sign(actuals[1:] - actuals[:-1])
        if len(preds) >= 2:
            pred_dir = np.sign(preds[1:] - actuals[:-1])
            actual_dir = np.sign(actuals[1:] - actuals[:-1])
            correct_dir = np.sum(pred_dir == actual_dir)
            total_dir = len(pred_dir)
            # Also check: was the direction of actual movement predicted?
            # i.e., did the model predict price would go up (down) when it actually went up (down)?
            dir_acc = correct_dir / total_dir if total_dir > 0 else np.nan
        else:
            dir_acc = np.nan

        summary[name] = {
            'wf_r2': wf_r2, 'wf_mae': wf_mae, 'wf_mape': wf_mape,
            'train_r2': train_r2, 'overfit': overfit,
            'dir_acc': dir_acc,
            'preds': preds, 'actuals': actuals,
            'dates': wf_results[name]['dates'],
            'n_rounds': len(preds)
        }
    return summary


# ---- 日线 Walk-Forward ----
print("\n  日线 Walk-Forward...")
y_d_valid = y_d.copy()
dts_d_all = dates_d.copy()
wf_d_results = {}
for name in daily_models:
    wf_d_results[name] = {'preds': [], 'actuals': [], 'dates': []}

wf_window_d = min(120, len(y_d_valid) - 10)
for train_end in range(wf_window_d, len(y_d_valid), step_size):
    train_start = train_end - wf_window_d
    X_train_d = X_d_scaled[train_start:train_end]
    y_train_d = y_d_valid[train_start:train_end]
    X_test_d = X_d_scaled[train_end:train_end+1]
    y_test_d = y_d_valid[train_end]
    date_test = dts_d_all[train_end]

    for name in daily_models:
        if name == 'ElasticNet':
            m = ElasticNet(alpha=0.1, l1_ratio=1.0, max_iter=20000, random_state=42)
        elif name == 'Ridge':
            m = Ridge(alpha=1.0)
        elif name == 'RandomForest':
            m = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5,
                                      max_features=0.7, random_state=42, n_jobs=-1)
        elif name == 'XGBoost':
            from xgboost import XGBRegressor
            m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.7,
                              reg_alpha=0.1, reg_lambda=1.0,
                              random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_train_d, y_train_d)
        pred = m.predict(X_test_d)[0]
        wf_d_results[name]['preds'].append(pred)
        wf_d_results[name]['actuals'].append(y_test_d)
        wf_d_results[name]['dates'].append(date_test)

wf_daily_summary = compute_wf_summary(wf_d_results, daily_results)
best_wf_daily = max(wf_daily_summary, key=lambda k: wf_daily_summary[k]['wf_r2'])
print(f"  ✅ 日线 WF 最佳: {best_wf_daily} (WF R²={wf_daily_summary[best_wf_daily]['wf_r2']:.4f})")

# ---- 周线 Walk-Forward（每个 horizon）----
wf_weekly_summaries = {}
for h_name, h_cfg in horizons.items():
    target = h_cfg['target']
    valid_mask = ~np.isnan(df_w[target].values)
    X_h = X_w_all[valid_mask]
    y_h = df_w.loc[valid_mask, target].values
    dts_all = dates_w[valid_mask]
    factories_ref = list(weekly_models[h_name]['models'].keys())

    wf_results_h = {name: {'preds': [], 'actuals': [], 'dates': []} for name in factories_ref}
    wf_win = min(80, len(y_h) - 10)

    for train_end in range(wf_win, len(y_h), step_size):
        train_start = train_end - wf_win
        X_train_h = X_h[train_start:train_end]
        y_train_h = y_h[train_start:train_end]
        X_test_h = X_h[train_end:train_end+1]
        y_test_h = y_h[train_end]
        date_test = dts_all[train_end]

        for name in factories_ref:
            if name == 'ElasticNet':
                m = ElasticNet(alpha=16.4, l1_ratio=1.0, max_iter=20000, random_state=42)
            elif name == 'Ridge':
                m = Ridge(alpha=1.0)
            elif name == 'RandomForest':
                m = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=5,
                                          max_features=0.7, random_state=42, n_jobs=-1)
            elif name == 'XGBoost':
                from xgboost import XGBRegressor
                m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.7,
                                  reg_alpha=0.1, reg_lambda=1.0,
                                  random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_train_h, y_train_h)
            pred = m.predict(X_test_h)[0]
            wf_results_h[name]['preds'].append(pred)
            wf_results_h[name]['actuals'].append(y_test_h)
            wf_results_h[name]['dates'].append(date_test)

    wf_weekly_summaries[h_name] = compute_wf_summary(wf_results_h, weekly_results[h_name]['results'])
    best_h_wf = max(wf_weekly_summaries[h_name], key=lambda k: wf_weekly_summaries[h_name][k]['wf_r2'])
    print(f"  ✅ 周线 [{h_name}] WF 最佳: {best_h_wf} "
          f"(WF R²={wf_weekly_summaries[h_name][best_h_wf]['wf_r2']:.4f}, "
          f"方向胜率={wf_weekly_summaries[h_name][best_h_wf]['dir_acc']:.1%})")


# ============================================================
#  PART 12：次日预测（滚动递归，日线）
# ============================================================
print("\n" + "=" * 70)
print("  次日价格预测（滚动递归 5 天）")
print("=" * 70)

n_pred_daily = 5
buffer_d_len = 60
last_date_d = df_d_v2['date'].iloc[-1]
last_price_d = df_d_v2['close'].iloc[-1]
future_dates_d = [last_date_d + pd.Timedelta(days=i+1) for i in range(n_pred_daily)]

buf_d = df_d_v2.iloc[-buffer_d_len:].copy().reset_index(drop=True)
predictions_daily = []

for step in range(n_pred_daily):
    b = buf_d.copy()
    n = len(b)
    feat = {}
    feat['current_price'] = b['close'].iloc[-1]

    feat['ret_d1'] = (b['close'].iloc[-1] - b['close'].iloc[-2]) / b['close'].iloc[-2] if n >= 2 else 0
    feat['ret_d5'] = (b['close'].iloc[-1] - b['close'].iloc[-6]) / b['close'].iloc[-6] if n >= 6 else feat['ret_d1']
    feat['ret_d10'] = (b['close'].iloc[-1] - b['close'].iloc[-11]) / b['close'].iloc[-11] if n >= 11 else feat['ret_d5']
    feat['ret_d20'] = (b['close'].iloc[-1] - b['close'].iloc[-21]) / b['close'].iloc[-21] if n >= 21 else feat['ret_d10']

    feat['vol_d5'] = b['close'].pct_change().iloc[-5:].std() if n >= 6 else b['close'].pct_change().std()
    feat['vol_d10'] = b['close'].pct_change().iloc[-10:].std() if n >= 11 else feat['vol_d5']
    feat['vol_d20'] = b['close'].pct_change().iloc[-20:].std() if n >= 21 else feat['vol_d10']

    ma5 = b['close'].iloc[-5:].mean() if n >= 5 else b['close'].iloc[-1]
    ma10 = b['close'].iloc[-10:].mean() if n >= 10 else ma5
    ma20 = b['close'].iloc[-20:].mean() if n >= 20 else ma10
    ma60 = b['close'].iloc[-60:].mean() if n >= 60 else ma20
    close_cur = b['close'].iloc[-1]
    feat['bias_ma5'] = (close_cur - ma5) / ma5 if ma5 != 0 else 0
    feat['bias_ma20'] = (close_cur - ma20) / ma20 if ma20 != 0 else 0
    feat['bias_ma60'] = (close_cur - ma60) / ma60 if ma60 != 0 else 0

    std20 = b['close'].iloc[-20:].std() if n >= 20 else b['close'].std()
    feat['band_width'] = 4 * std20 / ma20 if ma20 != 0 else 0.02
    feat['band_pos'] = (close_cur - (ma20 - 2 * std20)) / (4 * std20 + 1e-10) if n >= 20 else 0.5

    delta = b['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat['rsi_14'] = 100 - 100 / (1 + rs).iloc[-1] if n >= 15 else 50

    vol_ma5 = b['volume'].iloc[-5:].mean() if n >= 5 else b['volume'].iloc[-1]
    feat['vol_ratio_d'] = b['volume'].iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1
    feat['oi_chg_d1'] = (b['oi'].iloc[-1] - b['oi'].iloc[-2]) / b['oi'].iloc[-2] if n >= 2 and b['oi'].iloc[-2] != 0 else 0
    feat['oi_chg_d5'] = (b['oi'].iloc[-1] - b['oi'].iloc[-6]) / b['oi'].iloc[-6] if n >= 6 and b['oi'].iloc[-6] != 0 else 0
    feat['range_pct_d'] = (b['high'].iloc[-1] - b['low'].iloc[-1]) / close_cur

    ema12_val = b['close'].ewm(span=12).mean().iloc[-1]
    ema26_val = b['close'].ewm(span=26).mean().iloc[-1]
    feat['macd'] = ema12_val - ema26_val
    feat['macd_signal'] = b['close'].ewm(span=9).mean().iloc[-1] - ema26_val
    feat['macd_hist'] = feat['macd'] - feat['macd_signal']

    if has_wheat_d and 'wheat_profit' in b.columns:
        wp = b['wheat_profit'].dropna()
        feat['wheat_profit'] = wp.iloc[-1] if len(wp) > 0 else 0
        feat['wheat_ma4'] = wp.iloc[-4:].mean() if len(wp) >= 4 else feat.get('wheat_profit', 0)
    else:
        if has_wheat_d:
            feat['wheat_profit'] = 0
            feat['wheat_ma4'] = 0

    if 'settle_spread' in daily_features:
        feat['settle_spread'] = b['settle'].iloc[-1] - close_cur if pd.notna(b['settle'].iloc[-1]) else 0

    feat_vec_d = np.array([feat[c] for c in daily_features]).reshape(1, -1)
    feat_scaled_d = scaler_d.transform(feat_vec_d)
    pred_price = best_daily_model.predict(feat_scaled_d)[0]
    predictions_daily.append(pred_price)

    new_row = b.iloc[-1:].copy()
    new_row['date'] = future_dates_d[step]
    new_row['close'] = pred_price
    new_row['open'] = pred_price
    new_row['high'] = pred_price * (1 + abs(feat['range_pct_d']) * 0.5)
    new_row['low'] = pred_price * (1 - abs(feat['range_pct_d']) * 0.5)
    new_row['settle'] = pred_price
    new_row['volume'] = b['volume'].iloc[-1]
    new_row['oi'] = b['oi'].iloc[-1]
    buf_d = pd.concat([buf_d, new_row], ignore_index=True)

predictions_daily = np.array(predictions_daily)
d_mae = daily_results[best_daily_name]['resid_std']
d_spread = np.linspace(0.5, 1.2, n_pred_daily) * d_mae


# ============================================================
#  PART 13：周线预测（各时间尺度）
# ============================================================
print("\n" + "=" * 70)
print("  周线多时间尺度预测（1周/2周/4周/8周）")
print("=" * 70)

weekly_predictions = {}
buffer_w_len = 30
last_date_w = df_w['date'].iloc[-1]
last_price_w = df_w['close'].iloc[-1]

for h_name, h_cfg in horizons.items():
    best_model_name = weekly_results[h_name]['best']
    best_model = weekly_models[h_name]['models'][best_model_name]
    h_mae = weekly_results[h_name]['best_result']['resid_std']
    n_step = h_cfg['n_step']

    buf_w = df_w.iloc[-buffer_w_len:].copy().reset_index(drop=True)
    preds = []

    for step in range(n_step):
        b = buf_w.copy()
        n = len(b)
        feat = {}
        feat['current_price'] = b['close'].iloc[-1]

        feat['ret_1w'] = (b['close'].iloc[-1] - b['close'].iloc[-2]) / b['close'].iloc[-2] if n >= 2 else 0
        feat['ret_2w'] = (b['close'].iloc[-1] - b['close'].iloc[-3]) / b['close'].iloc[-3] if n >= 3 else feat['ret_1w']
        feat['ret_4w'] = (b['close'].iloc[-1] - b['close'].iloc[-5]) / b['close'].iloc[-5] if n >= 5 else feat['ret_2w']
        feat['ret_8w'] = (b['close'].iloc[-1] - b['close'].iloc[-9]) / b['close'].iloc[-9] if n >= 9 else feat['ret_4w']
        feat['mom_diff'] = feat['ret_1w'] - feat['ret_4w']
        feat['mom_accel'] = feat['ret_1w'] - 2 * feat['ret_4w'] + feat['ret_8w']

        ma5 = b['close'].iloc[-5:].mean() if n >= 5 else b['close'].iloc[-1]
        ma10 = b['close'].iloc[-10:].mean() if n >= 10 else ma5
        ma20 = b['close'].iloc[-20:].mean() if n >= 20 else ma10
        close_cur = b['close'].iloc[-1]
        feat['bias_5'] = (close_cur - ma5) / ma5 if ma5 != 0 else 0
        feat['bias_10'] = (close_cur - ma10) / ma10 if ma10 != 0 else 0
        feat['bias_20'] = (close_cur - ma20) / ma20 if ma20 != 0 else 0

        feat['vol_4w'] = b['close'].pct_change().iloc[-4:].std() if n >= 5 else b['close'].pct_change().std()
        feat['vol_8w'] = b['close'].pct_change().iloc[-8:].std() if n >= 9 else feat['vol_4w']
        feat['vol_16w'] = b['close'].pct_change().iloc[-16:].std() if n >= 17 else feat['vol_8w']
        feat['range_pct'] = (b['high'].iloc[-1] - b['low'].iloc[-1]) / close_cur

        vol_ma5 = b['volume'].iloc[-5:].mean() if n >= 5 else b['volume'].iloc[-1]
        feat['vol_ratio'] = b['volume'].iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1
        feat['oi_chg'] = (b['oi'].iloc[-1] - b['oi'].iloc[-2]) / b['oi'].iloc[-2] if n >= 2 and b['oi'].iloc[-2] != 0 else 0
        feat['oi_chg_4w'] = (b['oi'].iloc[-1] - b['oi'].iloc[-5]) / b['oi'].iloc[-5] if n >= 5 and b['oi'].iloc[-5] != 0 else 0
        feat['oi_chg_8w'] = (b['oi'].iloc[-1] - b['oi'].iloc[-9]) / b['oi'].iloc[-9] if n >= 9 and b['oi'].iloc[-9] != 0 else 0

        ema12_val = b['close'].ewm(span=12).mean().iloc[-1]
        ema26_val = b['close'].ewm(span=26).mean().iloc[-1]
        feat['macd'] = ema12_val - ema26_val
        feat['macd_signal'] = b['close'].ewm(span=9).mean().iloc[-1] - ema26_val
        feat['macd_hist'] = feat['macd'] - feat['macd_signal']

        if has_wheat:
            wp = b['wheat_profit'].dropna()
            feat['wheat_profit'] = wp.iloc[-1] if len(wp) > 0 else 0
            feat['wheat_profit_ma4'] = wp.iloc[-4:].mean() if len(wp) >= 4 else feat.get('wheat_profit', 0)
            feat['wheat_trend'] = (wp.iloc[-1] - wp.iloc[-5]) / wp.iloc[-5] if len(wp) >= 5 else 0
        else:
            feat['wheat_profit'] = 0
            feat['wheat_profit_ma4'] = 0
            feat['wheat_trend'] = 0

        if 'settle_spread' in weekly_features:
            feat['settle_spread'] = b['settle'].iloc[-1] - close_cur if pd.notna(b['settle'].iloc[-1]) else 0

        feat_vec_w = np.array([feat[c] for c in weekly_features]).reshape(1, -1)
        feat_scaled_w = scaler_w.transform(feat_vec_w)
        pred_price = best_model.predict(feat_scaled_w)[0]
        preds.append(pred_price)

        next_date = last_date_w + pd.Timedelta(weeks=step + 1)
        new_row = b.iloc[-1:].copy()
        new_row['date'] = next_date
        new_row['close'] = pred_price
        new_row['open'] = pred_price
        new_row['high'] = pred_price * (1 + abs(feat['range_pct']) * 0.5)
        new_row['low'] = pred_price * (1 - abs(feat['range_pct']) * 0.5)
        new_row['settle'] = pred_price
        new_row['volume'] = b['volume'].iloc[-1]
        new_row['oi'] = b['oi'].iloc[-1]
        new_row['year_week'] = f"{next_date.isocalendar()[0]}_{str(next_date.isocalendar()[1]).zfill(2)}"
        buf_w = pd.concat([buf_w, new_row], ignore_index=True)

    weekly_predictions[h_name] = {
        'predictions': np.array(preds),
        'final_pred': preds[-1],
        'final_date': last_date_w + pd.Timedelta(weeks=n_step),
        'mae': h_mae,
        'best_model': best_model_name,
        'wf_r2': wf_weekly_summaries.get(h_name, {}).get(best_model_name, {}).get('wf_r2', np.nan),
        'wf_mae': wf_weekly_summaries.get(h_name, {}).get(best_model_name, {}).get('wf_mae', np.nan),
        'dir_acc': wf_weekly_summaries.get(h_name, {}).get(best_model_name, {}).get('dir_acc', np.nan),
        'spread': np.linspace(0.4, 1.2, n_step) * h_mae,
    }

    print(f"\n  [{h_name}] 最终预测: {preds[-1]:.2f} 元/吨 "
          f"(日期: {weekly_predictions[h_name]['final_date'].strftime('%Y-%m-%d')})")
    print(f"       模型: {best_model_name} | WF R²={weekly_predictions[h_name]['wf_r2']:.4f} "
          f"| 方向胜率: {weekly_predictions[h_name]['dir_acc']:.1%}")


# ============================================================
#  PART 14：趋势强度 & 方向概率分析
# ============================================================
print("\n" + "=" * 70)
print("  趋势强度 & 方向概率分析")
print("=" * 70)

# 基于周线数据计算趋势
trend_score, score_detail = compute_trend_strength(df_w, n_weeks=8)
trend_name, trend_color = trend_label(trend_score)

print(f"\n  综合趋势强度: {trend_score:.1f}/100  → {trend_name}")
for name, sc, w in score_detail:
    bar = '█' * int(sc / 10) + '░' * (10 - int(sc / 10))
    print(f"    {name:<4} {bar} {sc:.0f} (权重{w:.0%})")

# 日线方向概率 — 构建完整特征向量（与训练时的 daily_features 完全一致）
d_feat_full = {}
b_d = df_d_v2.iloc[-60:].copy().reset_index(drop=True)
n_d = len(b_d)
close_cur = b_d['close'].iloc[-1]

# 基本收益率特征
d_feat_full['ret_d1'] = (b_d['close'].iloc[-1] - b_d['close'].iloc[-2]) / b_d['close'].iloc[-2] if n_d >= 2 else 0
d_feat_full['ret_d5'] = (b_d['close'].iloc[-1] - b_d['close'].iloc[-6]) / b_d['close'].iloc[-6] if n_d >= 6 else d_feat_full['ret_d1']
d_feat_full['ret_d10'] = (b_d['close'].iloc[-1] - b_d['close'].iloc[-11]) / b_d['close'].iloc[-11] if n_d >= 11 else d_feat_full['ret_d5']
d_feat_full['ret_d20'] = (b_d['close'].iloc[-1] - b_d['close'].iloc[-21]) / b_d['close'].iloc[-21] if n_d >= 21 else d_feat_full['ret_d10']

# 波动率
d_feat_full['vol_d5'] = b_d['close'].pct_change().iloc[-5:].std() if n_d >= 6 else b_d['close'].pct_change().std()
d_feat_full['vol_d10'] = b_d['close'].pct_change().iloc[-10:].std() if n_d >= 11 else d_feat_full['vol_d5']
d_feat_full['vol_d20'] = b_d['close'].pct_change().iloc[-20:].std() if n_d >= 21 else d_feat_full['vol_d10']

# 均线
ma5 = b_d['close'].iloc[-5:].mean() if n_d >= 5 else b_d['close'].iloc[-1]
ma10 = b_d['close'].iloc[-10:].mean() if n_d >= 10 else ma5
ma20 = b_d['close'].iloc[-20:].mean() if n_d >= 20 else ma10
ma60 = b_d['close'].iloc[-60:].mean() if n_d >= 60 else ma20
d_feat_full['bias_ma5'] = (close_cur - ma5) / ma5 if ma5 != 0 else 0
d_feat_full['bias_ma20'] = (close_cur - ma20) / ma20 if ma20 != 0 else 0
d_feat_full['bias_ma60'] = (close_cur - ma60) / ma60 if ma60 != 0 else 0

# 布林带
std20 = b_d['close'].iloc[-20:].std() if n_d >= 20 else b_d['close'].std()
d_feat_full['band_width'] = 4 * std20 / ma20 if ma20 != 0 else 0.02
d_feat_full['band_pos'] = (close_cur - (ma20 - 2 * std20)) / (4 * std20 + 1e-10) if n_d >= 20 else 0.5

# RSI
d_feat_full['rsi_14'] = float(compute_rsi(b_d['close'], 14).iloc[-1]) if n_d >= 15 else 50.0

# 量和持仓
vol_ma5_d = b_d['volume'].iloc[-5:].mean() if n_d >= 5 else b_d['volume'].iloc[-1]
d_feat_full['vol_ratio_d'] = b_d['volume'].iloc[-1] / vol_ma5_d if vol_ma5_d > 0 else 1.0
d_feat_full['oi_chg_d1'] = (b_d['oi'].iloc[-1] - b_d['oi'].iloc[-2]) / b_d['oi'].iloc[-2] if n_d >= 2 and b_d['oi'].iloc[-2] != 0 else 0.0
d_feat_full['oi_chg_d5'] = (b_d['oi'].iloc[-1] - b_d['oi'].iloc[-6]) / b_d['oi'].iloc[-6] if n_d >= 6 and b_d['oi'].iloc[-6] != 0 else 0.0
d_feat_full['range_pct_d'] = (b_d['high'].iloc[-1] - b_d['low'].iloc[-1]) / close_cur

# MACD
ema12_v = b_d['close'].ewm(span=12).mean().iloc[-1]
ema26_v = b_d['close'].ewm(span=26).mean().iloc[-1]
d_feat_full['macd'] = ema12_v - ema26_v
d_feat_full['macd_signal'] = b_d['close'].ewm(span=9).mean().iloc[-1] - ema26_v
d_feat_full['macd_hist'] = d_feat_full['macd'] - d_feat_full['macd_signal']

# 小麦和结算
if has_wheat_d and 'wheat_profit' in b_d.columns:
    wp = b_d['wheat_profit'].dropna()
    d_feat_full['wheat_profit'] = float(wp.iloc[-1]) if len(wp) > 0 else 0.0
    d_feat_full['wheat_ma4'] = float(wp.iloc[-4:].mean()) if len(wp) >= 4 else d_feat_full.get('wheat_profit', 0.0)

if 'settle_spread' in daily_features:
    d_feat_full['settle_spread'] = b_d['settle'].iloc[-1] - close_cur if pd.notna(b_d['settle'].iloc[-1]) else 0.0

d_prob = directional_probability_monte_carlo(
    best_daily_model, scaler_d, d_feat_full, n_simulations=5000
)
print(f"\n  次日价格概率分布:")
print(f"    基准价: {d_prob['base_pred']:.2f}")
print(f"    上涨概率 (>+0.5%): {d_prob['prob_up']:.1%}")
print(f"    震荡概率:          {d_prob['prob_flat']:.1%}")
print(f"    下跌概率 (<-0.5%): {d_prob['prob_down']:.1%}")
print(f"    90%置信区间: [{d_prob['p5']:.0f}, {d_prob['p95']:.0f}]")


# ============================================================
#  PART 15：输出汇总表格
# ============================================================
print("\n" + "=" * 80)
print("  多时间尺度预测汇总")
print("=" * 80)

# 日线
print(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
print(f"  │                    【次日~5日】 日线预测                              │")
print(f"  ├───────┬──────────┬───────────┬───────────┬────────────────────────────┤")
print(f"  │  日期  │  预测价  │  90%CI下限 │  90%CI上限 │  涨跌                     │")
print(f"  ├───────┼──────────┼───────────┼───────────┼────────────────────────────┤")
for i in range(n_pred_daily):
    dt = future_dates_d[i].strftime('%m-%d')
    p = predictions_daily[i]
    lo = p - d_spread[i] * 1.65
    hi = p + d_spread[i] * 1.65
    chg = predictions_daily[i] - last_price_d if i == 0 else predictions_daily[i] - predictions_daily[i-1]
    chg_pct = chg / (last_price_d if i == 0 else predictions_daily[i-1]) * 100
    arrow = '↑' if chg > 0 else '↓' if chg < 0 else '→'
    print(f"  │ 第{i+1}天  │  {p:>6.1f}  │   {lo:>6.1f}  │   {hi:>6.1f}  │ {arrow} {chg:>+5.1f}({chg_pct:>+5.1f}%)     │")
print(f"  └───────┴──────────┴───────────┴───────────┴────────────────────────────┘")

# 周线
for h_name, h_cfg in horizons.items():
    wp = weekly_predictions[h_name]
    final_pred = wp['final_pred']
    final_date = wp['final_date']
    chg = final_pred - last_price_w
    chg_pct = chg / last_price_w * 100
    conf_lo = final_pred - wp['spread'][-1] * 1.65
    conf_hi = final_pred + wp['spread'][-1] * 1.65

    horizon_labels = {'1w': '【1周】', '2w': '【2周】', '4w': '【4周】', '8w': '【约半月~2月】'}
    actual_last_date = last_price_d  # use 5-12 daily close as reference
    print(f"\n  ┌─────────────────────────────────────────────────────────────────────┐")
    print(f"  │  {horizon_labels[h_name]} 周线预测  ⚠️数据截止: 2026-05-12                  │")
    print(f"  ├─────────────────────────────────────────────────────────────────────┤")
    print(f"  │  基准价: {actual_last_date:.2f} 元/吨  (2026-05-12 收盘)                       │")
    print(f"  │  预测价: {final_pred:.2f} 元/吨  (周线模型预测区间)                       │")
    print(f"  │  预测涨跌: {chg:>+7.2f} 元 ({chg_pct:>+6.1f}%)                                    │")
    print(f"  │  90%置信区间: [{conf_lo:.1f}, {conf_hi:.1f}]                                │")
    print(f"  │  最优模型: {wp['best_model']}  WF R²={wp['wf_r2']:.4f}  方向胜率={wp['dir_acc']:.1%}      │")
    print(f"  └─────────────────────────────────────────────────────────────────────┘")


# ============================================================
#  PART 16：可视化 — 多时间尺度预测总览图
# ============================================================
fig = plt.figure(figsize=(22, 18))
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.30)

# ── 图1: 全局多时间尺度预测曲线 ──
ax1 = fig.add_subplot(gs[0, :])
n_hist_w = min(60, len(df_w))
hd_w = df_w['date'].iloc[-n_hist_w:].values
hp_w = df_w['close'].iloc[-n_hist_w:].values

ax1.plot(hd_w, hp_w, 'k-', lw=2.5, label='历史收盘价', zorder=10)
ax1.axvline(x=last_date_w, color='#FF5722', ls='--', lw=2, alpha=.7, label='预测起点')

# 周线预测
h_colors = {'1w': '#E91E63', '2w': '#FF9800', '4w': '#2196F3', '8w': '#4CAF50'}
h_labels = {'1w': '1周', '2w': '2周', '4w': '4周', '8w': '约半月~2月'}
for h_name in horizons:
    wp = weekly_predictions[h_name]
    n_step = horizons[h_name]['n_step']
    future_w_dates = [last_date_w + pd.Timedelta(weeks=i+1) for i in range(n_step)]
    conf_lo = wp['predictions'] - wp['spread'] * 1.65
    conf_hi = wp['predictions'] + wp['spread'] * 1.65
    ax1.plot(future_w_dates, wp['predictions'], color=h_colors[h_name], lw=2.5,
             marker='D', ms=6, label=f"{h_labels[h_name]} 预测 ({wp['best_model']}, R²={wp['wf_r2']:.2f})")
    ax1.fill_between(future_w_dates, conf_lo, conf_hi, alpha=.12, color=h_colors[h_name])
    ax1.annotate(f"{wp['final_pred']:.0f}", xy=(future_w_dates[-1], wp['final_pred']),
                 xytext=(5, 0), textcoords='offset points', fontsize=9, fontweight='bold',
                 color=h_colors[h_name])

ax1.set_title('多时间尺度预测总览 — 玉米加权期货 | 数据截止: 2026-05-12', fontsize=15, fontweight='bold')
ax1.set_ylabel('价格 (元/吨)', fontsize=12)
ax1.legend(fontsize=9, loc='upper left', ncol=3)
ax1.grid(True, alpha=.3)

# ── 图2: 趋势强度仪表盘 ──
ax2 = fig.add_subplot(gs[1, 0])
theta = np.linspace(0, np.pi, 500)
r_outer = 1.0
x_outer = r_outer * np.cos(theta)
y_outer = r_outer * np.sin(theta)
ax2.fill_between(x_outer, 0, y_outer, where=(y_outer >= 0), color='#D32F2F', alpha=.3)
ax2.fill_between(x_outer, 0, y_outer, where=(y_outer >= 0), color='#D32F2F', alpha=.1)

arc_theta = np.linspace(0, np.pi * (trend_score / 100), 200)
arc_r = 0.85
ax2.plot(arc_r * np.cos(np.linspace(0, np.pi, 500)),
         arc_r * np.sin(np.linspace(0, np.pi, 500)), 'k-', lw=1, alpha=.2)
ax2.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta),
         color=trend_color, lw=8, solid_capstyle='round')

ax2.annotate(f'{trend_score:.0f}', xy=(0, 0.15), fontsize=36, ha='center', va='center',
             fontweight='bold', color=trend_color)
ax2.annotate(trend_name, xy=(0, -0.08), fontsize=14, ha='center', va='center',
             fontweight='bold', color=trend_color)

ax2.set_xlim(-1.3, 1.3)
ax2.set_ylim(-0.3, 1.2)
ax2.axis('off')
ax2.set_title('趋势强度综合评分', fontsize=13, fontweight='bold')

# ── 图3: 方向概率饼图 ──
ax3 = fig.add_subplot(gs[1, 1])
prob_labels = ['上涨\n(>+0.5%)', '震荡', '下跌\n(<-0.5%)']
prob_vals = [d_prob['prob_up'], d_prob['prob_flat'], d_prob['prob_down']]
prob_colors = ['#4CAF50', '#FFC107', '#F44336']
wedges, texts, autotexts = ax3.pie(
    prob_vals, labels=prob_labels, colors=prob_colors,
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'lw': 2}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
ax3.set_title(f'次日涨跌方向概率分布\n(基准: {d_prob["base_pred"]:.0f} 元/吨)', fontsize=13, fontweight='bold')

# ── 图4: 各时间尺度预测价格对比 ──
ax4 = fig.add_subplot(gs[2, 0])
h_labels_short = {'1w': '1周', '2w': '2周', '4w': '4周', '8w': '约半月'}
h_vals = [weekly_predictions[h]['final_pred'] for h in horizons]
h_err_lo = [weekly_predictions[h]['final_pred'] - (weekly_predictions[h]['predictions'][-1] - weekly_predictions[h]['spread'][-1] * 1.65)
              for h in horizons]
h_err_hi = [(weekly_predictions[h]['predictions'][-1] + weekly_predictions[h]['spread'][-1] * 1.65) - weekly_predictions[h]['final_pred']
              for h in horizons]

bars = ax4.bar(range(len(horizons)), h_vals, color=[h_colors[h] for h in horizons],
               edgecolor='black', lw=.5, alpha=.85, yerr=[h_err_lo, h_err_hi], capsize=5)
ax4.axhline(last_price_w, color='red', ls='--', lw=2, label=f'当前价: {last_price_w:.0f}')
ax4.set_xticks(range(len(horizons)))
ax4.set_xticklabels([h_labels_short[h] for h in horizons], fontsize=12)
ax4.set_ylabel('预测价格 (元/吨)', fontsize=12)
ax4.set_title('各时间尺度最终预测价格', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=.3, axis='y')
for bar, val in zip(bars, h_vals):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
             f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ── 图5: 各时间尺度涨跌率 ──
ax5 = fig.add_subplot(gs[2, 1])
cum_chgs = [(weekly_predictions[h]['final_pred'] - last_price_w) / last_price_w * 100
            for h in horizons]
bar_colors = ['#4CAF50' if c > 0 else '#F44336' for c in cum_chgs]
bars2 = ax5.bar(range(len(horizons)), cum_chgs, color=bar_colors,
                edgecolor='black', lw=.5, alpha=.85)
ax5.axhline(0, color='black', lw=1)
ax5.set_xticks(range(len(horizons)))
ax5.set_xticklabels([h_labels_short[h] for h in horizons], fontsize=12)
ax5.set_ylabel('预测涨跌 (%)', fontsize=12)
ax5.set_title('各时间尺度预测涨跌幅度', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=.3, axis='y')
for bar, val in zip(bars2, cum_chgs):
    ax5.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2 if val > 0 else bar.get_height() - 0.5,
             f'{val:+.1f}%', ha='center', fontsize=10, fontweight='bold')

# ── 图6: Walk-Forward R² 热力图 ──
ax6 = fig.add_subplot(gs[3, 0])
model_names_sorted = list(factories.keys())
horizon_names = list(horizons.keys())
wf_r2_matrix = np.zeros((len(horizon_names), len(model_names_sorted)))
wf_mae_matrix = np.zeros_like(wf_r2_matrix)

for hi, h_name in enumerate(horizon_names):
    for mi, m_name in enumerate(model_names_sorted):
        if m_name in wf_weekly_summaries.get(h_name, {}):
            wf_r2_matrix[hi, mi] = wf_weekly_summaries[h_name][m_name]['wf_r2']
            wf_mae_matrix[hi, mi] = wf_weekly_summaries[h_name][m_name]['wf_mae']

im = ax6.imshow(wf_r2_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.2, vmax=1.0)
ax6.set_xticks(range(len(model_names_sorted)))
ax6.set_xticklabels(model_names_sorted, fontsize=11)
ax6.set_yticks(range(len(horizon_names)))
ax6.set_yticklabels([h_labels_short[h] for h in horizon_names], fontsize=11)
ax6.set_title('Walk-Forward R² 热力图（按时间尺度 × 模型）', fontsize=13, fontweight='bold')
for i in range(len(horizon_names)):
    for j in range(len(model_names_sorted)):
        val = wf_r2_matrix[i, j]
        ax6.text(j, i, f'{val:.2f}', ha='center', va='center',
                 fontsize=10, fontweight='bold',
                 color='white' if val < 0.3 or val > 0.7 else 'black')
plt.colorbar(im, ax=ax6, label='WF R²')

# ── 图7: Walk-Forward 方向胜率 ──
ax7 = fig.add_subplot(gs[3, 1])
dir_acc_matrix = np.zeros((len(horizon_names), len(model_names_sorted)))
for hi, h_name in enumerate(horizon_names):
    for mi, m_name in enumerate(model_names_sorted):
        if m_name in wf_weekly_summaries.get(h_name, {}):
            dir_acc_matrix[hi, mi] = wf_weekly_summaries[h_name][m_name]['dir_acc']

im2 = ax7.imshow(dir_acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax7.set_xticks(range(len(model_names_sorted)))
ax7.set_xticklabels(model_names_sorted, fontsize=11)
ax7.set_yticks(range(len(horizon_names)))
ax7.set_yticklabels([h_labels_short[h] for h in horizon_names], fontsize=11)
ax7.set_title('Walk-Forward 方向胜率热力图', fontsize=13, fontweight='bold')
for i in range(len(horizon_names)):
    for j in range(len(model_names_sorted)):
        val = dir_acc_matrix[i, j]
        if not np.isnan(val):
            ax7.text(j, i, f'{val:.0%}', ha='center', va='center',
                     fontsize=10, fontweight='bold',
                     color='white' if val < 0.5 else 'black')
plt.colorbar(im2, ax=ax7, label='方向胜率')

fig.suptitle(f'玉米期货多时间尺度预测分析 | 基准: {last_price_w:.0f} 元/吨 (数据截止 2026-05-12)',
             fontsize=16, fontweight='bold', y=1.01)

plt.savefig('v3_multi_horizon_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 多时间尺度预测总览图已保存: v3_multi_horizon_overview.png")


# ============================================================
#  PART 17：日线预测详细图
# ============================================================
fig2, axes2 = plt.subplots(2, 1, figsize=(18, 10), gridspec_kw={'height_ratios': [3, 1]})
ax = axes2[0]

n_hist_d = min(60, len(df_d_v2))
hd_d = df_d_v2['date'].iloc[-n_hist_d:].values
hp_d = df_d_v2['close'].iloc[-n_hist_d:].values

ax.plot(hd_d, hp_d, 'k-', lw=2, label='历史收盘价', zorder=10)
ax.axvline(x=last_date_d, color='#FF5722', ls='--', lw=2, alpha=.7, label='预测起点')

d_lo = predictions_daily - d_spread * 1.65
d_hi = predictions_daily + d_spread * 1.65
ax.plot(future_dates_d, predictions_daily, color='#E91E63', lw=2.5, marker='D', ms=7,
        label=f'次日预测 ({best_daily_name})')
ax.fill_between(future_dates_d, d_lo, d_hi, alpha=.20, color='#E91E63', label='90%置信区间')

for d, p in zip(future_dates_d, predictions_daily):
    ax.annotate(f'{p:.0f}', xy=(d, p), xytext=(0, 14),
                textcoords='offset points', ha='center',
                fontsize=10, fontweight='bold', color='#E91E63',
                arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1))

ax.set_title(f'日线预测 — 未来{n_pred_daily}天 ({best_daily_name}, WF R²={wf_daily_summary[best_wf_daily]["wf_r2"]:.4f})',
             fontsize=14, fontweight='bold')
ax.set_ylabel('价格 (元/吨)', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=.3)

ax2 = axes2[1]
cum_chg_d = predictions_daily - last_price_d
bar_colors_d = ['#4CAF50' if c > 0 else '#F44336' for c in cum_chg_d]
ax2.bar(range(n_pred_daily), cum_chg_d, color=bar_colors_d, edgecolor='black', lw=.5, alpha=.85)
ax2.set_xticks(range(n_pred_daily))
ax2.set_xticklabels([f'第{i+1}天\n{d.strftime("%m-%d")}' for i, d in enumerate(future_dates_d)])
ax2.axhline(0, color='black', lw=1)
ax2.set_ylabel('相对基准涨跌 (元)', fontsize=12)
ax2.set_title(f'每日预测涨跌（基准: {last_price_d:.0f} 元/吨）', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=.3, axis='y')

plt.tight_layout()
plt.savefig('v3_daily_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 日线预测图已保存: v3_daily_forecast.png")


# ============================================================
#  PART 18：各时间尺度独立预测图
# ============================================================
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 12))
axes3 = axes3.flatten()

for idx, (h_name, h_cfg) in enumerate(horizons.items()):
    ax = axes3[idx]
    wp = weekly_predictions[h_name]
    n_step = h_cfg['n_step']
    future_dates_h = [last_date_w + pd.Timedelta(weeks=i+1) for i in range(n_step)]

    n_hist_h = min(40, len(df_w))
    hd_h = df_w['date'].iloc[-n_hist_h:].values
    hp_h = df_w['close'].iloc[-n_hist_h:].values

    ax.plot(hd_h, hp_h, 'k-', lw=2, label='历史收盘')
    ax.axvline(x=last_date_w, color='#FF5722', ls='--', lw=1.5, alpha=.6)

    conf_lo_h = wp['predictions'] - wp['spread'] * 1.65
    conf_hi_h = wp['predictions'] + wp['spread'] * 1.65

    ax.plot(future_dates_h, wp['predictions'], color=h_colors[h_name], lw=2.5, marker='D', ms=7,
            label=f"预测 ({wp['best_model']})")
    ax.fill_between(future_dates_h, conf_lo_h, conf_hi_h, alpha=.18, color=h_colors[h_name])
    ax.axhline(last_price_w, color='gray', ls=':', lw=1, alpha=.5)

    for d, p in zip(future_dates_h, wp['predictions']):
        ax.annotate(f'{p:.0f}', xy=(d, p), xytext=(0, 10),
                    textcoords='offset points', ha='center', fontsize=9,
                    fontweight='bold', color=h_colors[h_name])

    chg = wp['final_pred'] - last_price_w
    chg_pct = chg / last_price_w * 100
    ax.set_title(f'{h_labels[h_name]} 预测 | {wp["best_model"]} WF R²={wp["wf_r2"]:.3f}\n'
                 f'预测涨跌: {chg:+.1f}元 ({chg_pct:+.1f}%)  方向胜率: {wp["dir_acc"]:.1%}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('价格 (元/吨)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=.3)

plt.suptitle(f'各时间尺度周线预测详情 | 基准: {last_price_w:.0f} 元/吨 (数据截止 2026-05-12)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('v3_weekly_horizons.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 周线各时间尺度预测图已保存: v3_weekly_horizons.png")


# ============================================================
#  PART 19：Monte Carlo 概率分布可视化
# ============================================================
fig4, axes4 = plt.subplots(1, 4, figsize=(20, 5))
for idx, (h_name, h_cfg) in enumerate(horizons.items()):
    ax = axes4[idx]
    wp = weekly_predictions[h_name]
    n_step = h_cfg['n_step']

    final_pred = wp['final_pred']
    h_mae = wp['mae']
    sigma = h_mae / final_pred if final_pred != 0 else 0.01

    sims = np.random.normal(final_pred, sigma * abs(final_pred), 5000)
    ax.hist(sims, bins=50, color=h_colors[h_name], alpha=.7, edgecolor='white', lw=.3, density=True)

    p5, p50, p95 = np.percentile(sims, 5), np.percentile(sims, 50), np.percentile(sims, 95)
    ax.axvline(final_pred, color='red', lw=2, ls='--', label=f'预测均值: {final_pred:.0f}')
    ax.axvline(p50, color='orange', lw=1.5, ls=':', label=f'中位数: {p50:.0f}')
    ax.axvline(p5, color='blue', lw=1.5, ls=':', alpha=.5)
    ax.axvline(p95, color='blue', lw=1.5, ls=':', alpha=.5, label=f'5%-95%: [{p5:.0f},{p95:.0f}]')
    ax.fill_betweenx([0, ax.get_ylim()[1] * 2], p5, p95, alpha=.08, color=h_colors[h_name])

    p_up = np.mean(sims > last_price_w * 1.005)
    p_down = np.mean(sims < last_price_w * 0.995)
    ax.set_title(f'{h_labels[h_name]}\n上涨:{p_up:.0%} 下跌:{p_down:.0%}', fontsize=12, fontweight='bold')
    ax.set_xlabel('价格 (元/吨)', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=.3)

plt.suptitle('各时间尺度 Monte Carlo 概率分布', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('v3_monte_carlo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Monte Carlo 概率分布图已保存: v3_monte_carlo.png")


# ============================================================
#  PART 20：最终汇总
# ============================================================
print("\n" + "=" * 80)
print("  最终汇总")
print("=" * 80)
print(f"\n  数据范围: {df_w['date'].min().strftime('%Y-%m-%d')} → 2026-05-12 (日线) / 2026-05-18周线聚合标签(实际5-12收盘)")
print(f"  最新收盘: {last_price_w:.2f} 元/吨 (2026-05-12)")
print(f"\n  趋势强度: {trend_score:.0f}/100  →  {trend_name}")
print(f"  次日方向概率: 上涨 {d_prob['prob_up']:.1%} | 震荡 {d_prob['prob_flat']:.1%} | 下跌 {d_prob['prob_down']:.1%}")
print(f"\n  预测汇总:")
for h_name in horizons:
    wp = weekly_predictions[h_name]
    chg = wp['final_pred'] - last_price_w
    chg_pct = chg / last_price_w * 100
    print(f"    {h_labels[h_name]:>12}: {wp['final_pred']:>7.2f} 元  "
          f"({chg:>+6.1f}元 {chg_pct:>+6.1f}%) | WF R²={wp['wf_r2']:.4f} | 方向胜率={wp['dir_acc']:.1%}")

print(f"\n  生成图表:")
print(f"    v3_multi_horizon_overview.png  多时间尺度预测总览 + 趋势仪表盘")
print(f"    v3_daily_forecast.png          日线预测详细图")
print(f"    v3_weekly_horizons.png         周线各时间尺度预测图")
print(f"    v3_monte_carlo.png             Monte Carlo 概率分布图")
print("=" * 80)
