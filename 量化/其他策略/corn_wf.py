"""
玉米期货价格预测 — 多模型对比 + Walk-Forward 验证版
对比 ElasticNet / Ridge / RandomForest / XGBoost
Walk-Forward 验证每个模型的真实泛化能力
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, ElasticNet, RidgeCV, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib
import os

warnings.filterwarnings('ignore')
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'SimHei', 'Songti SC', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

os.chdir('/root/ff/量化/其他策略')

# ============================================================
#  PART 1：读取数据 & 预处理
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

agg_dict = {
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
    'volume': 'sum', 'oi': 'last', 'settle': 'last',
}
if 'wheat_profit' in df.columns:
    agg_dict['wheat_profit'] = 'last'

df_w = df.resample('W-MON', on='date').agg(agg_dict).dropna(subset=['close']).reset_index()
df_w = df_w.sort_values('date').reset_index(drop=True)
df_w['year'] = df_w['date'].dt.isocalendar().year.astype(int)
df_w['week'] = df_w['date'].dt.isocalendar().week.astype(int)
df_w['year_week'] = df_w['year'].astype(str) + '_' + df_w['week'].astype(str).str.zfill(2)

print(f"周线数据: {df_w['date'].min().strftime('%Y-%m-%d')} → {df_w['date'].max().strftime('%Y-%m-%d')}  共 {len(df_w)} 周")

# ============================================================
#  PART 2：特征工程
# ============================================================
df_f = df_w.copy()

df_f['ret_1w'] = df_f['close'].pct_change(1)
df_f['ret_4w'] = df_f['close'].pct_change(4)
df_f['ret_8w'] = df_f['close'].pct_change(8)
df_f['mom_diff'] = df_f['ret_1w'] - df_f['ret_4w']

df_f['ma5'] = df_f['close'].rolling(5).mean()
df_f['ma10'] = df_f['close'].rolling(10).mean()
df_f['ma20'] = df_f['close'].rolling(20).mean()
df_f['bias_5'] = (df_f['close'] - df_f['ma5']) / df_f['ma5']
df_f['bias_20'] = (df_f['close'] - df_f['ma20']) / df_f['ma20']

df_f['vol_4w'] = df_f['ret_1w'].rolling(4).std()
df_f['vol_8w'] = df_f['ret_1w'].rolling(8).std()
df_f['range_pct'] = (df_f['high'] - df_f['low']) / df_f['close']

df_f['vol_ma5'] = df_f['volume'].rolling(5).mean()
df_f['vol_ratio'] = df_f['volume'] / df_f['vol_ma5']
df_f['oi_chg'] = df_f['oi'].pct_change(1)
df_f['oi_chg_4w'] = df_f['oi'].pct_change(4)

has_wheat = 'wheat_profit' in df_f.columns and df_f['wheat_profit'].notna().sum() > 20
if has_wheat:
    df_f['wheat_profit_ma4'] = df_f['wheat_profit'].rolling(4).mean()
    print("华北小麦替代利润已加入")

if 'settle' in df_f.columns:
    df_f['settle_spread'] = df_f['settle'] - df_f['close']

price_features = [
    'ret_1w', 'ret_4w', 'ret_8w', 'mom_diff',
    'bias_5', 'bias_20',
    'vol_4w', 'vol_8w', 'range_pct',
    'vol_ratio', 'oi_chg', 'oi_chg_4w',
]
wheat_features = ['wheat_profit', 'wheat_profit_ma4'] if has_wheat else []
extra_features = ['settle_spread'] if 'settle_spread' in df_f.columns else []
all_features = price_features + wheat_features + extra_features
target_col = 'close'

print(f"特征数: {len(all_features)}")
df_v2 = df_f.dropna(subset=all_features + [target_col]).reset_index(drop=True)
print(f"样本: {len(df_v2)} 周  ({df_v2['date'].min().strftime('%Y-%m-%d')} → {df_v2['date'].max().strftime('%Y-%m-%d')})")

X = df_v2[all_features].values
y = df_v2[target_col].values
dates_all = df_v2['date'].values

# ============================================================
#  PART 3：标准化
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
#  PART 4：定义模型工厂
# ============================================================
def build_elasticnet(Xs, y_):
    m = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                     n_alphas=100, cv=5, max_iter=20000, random_state=42, n_jobs=-1)
    m.fit(Xs, y_)
    model = ElasticNet(alpha=m.alpha_, l1_ratio=m.l1_ratio_,
                       max_iter=20000, random_state=42)
    model.fit(Xs, y_)
    return model, 'ElasticNet'

def build_ridge(Xs, y_):
    m = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
    m.fit(Xs, y_)
    return m, 'Ridge'

def build_rf(Xs, y_):
    m = RandomForestRegressor(
        n_estimators=100, max_depth=6, min_samples_leaf=5,
        max_features=0.7, random_state=42, n_jobs=-1
    )
    m.fit(Xs, y_)
    return m, 'RandomForest'

def build_xgb(Xs, y_):
    from xgboost import XGBRegressor
    m = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    m.fit(Xs, y_)
    return m, 'XGBoost'

# ============================================================
#  PART 5：全量训练 & 评估
# ============================================================
print("\n" + "=" * 70)
print("  PART A — 全量训练（训练集 R²）")
print("=" * 70)

model_factories = [
    lambda: build_elasticnet(X_scaled, y),
    lambda: build_ridge(X_scaled, y),
    lambda: build_rf(X_scaled, y),
    lambda: build_xgb(X_scaled, y),
]
colors = {'ElasticNet': '#E91E63', 'Ridge': '#FF9800', 'RandomForest': '#2196F3', 'XGBoost': '#4CAF50'}

results_train = {}
models_trained = {}

for factory in model_factories:
    try:
        model, name = factory()
        yp = model.predict(X_scaled)
        r2 = r2_score(y, yp)
        mae = mean_absolute_error(y, yp)
        mape = mean_absolute_percentage_error(y, yp) * 100
        resid_std = np.std(y - yp)
        print(f"{name:<15} 训练R²={r2:.4f}  MAE={mae:.2f}  MAPE={mape:.2f}%  σ=±{resid_std:.2f}")
        results_train[name] = {'r2': r2, 'mae': mae, 'mape': mape, 'resid_std': resid_std, 'y_pred': yp}
        models_trained[name] = model
    except Exception as e:
        print(f"⚠️  {factory.__name__} 失败: {e}")

# ============================================================
#  PART 6：Walk-Forward 验证（核心！）
# ============================================================
print("\n" + "=" * 70)
print("  PART B — Walk-Forward 验证（真实泛化能力）")
print("=" * 70)
print("  策略: 滚动窗口，每次用过去 N 周训练，预测下 1 周")
print("  训练窗口: 80 周  |  步长: 10 周")

train_window = 80
step_size = 10  # 每4周做一次预测，大幅加速
min_train = train_window
n_total = len(df_v2)

wf_results = {name: {'preds': [], 'actuals': [], 'dates': []} for name in models_trained}

for train_end in range(min_train, n_total, step_size):
    train_start = train_end - train_window
    test_idx = train_end

    X_train = X_scaled[train_start:train_end]
    y_train = y[train_start:train_end]
    X_test = X_scaled[test_idx:test_idx+1]
    y_test = y[test_idx]

    model_names = list(wf_results.keys())
    for name in model_names:
        if name == 'ElasticNet':
            # 用固定参数避免每次循环跑 ElasticNetCV
            m = ElasticNet(alpha=16.4, l1_ratio=1.0, max_iter=20000, random_state=42)
        elif name == 'Ridge':
            # 用固定 alpha=1 避免每次跑 RidgeCV
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
        wf_results[name]['dates'].append(dates_all[test_idx])

print(f"\n  Walk-Forward 共完成 {len(wf_results[list(models_trained.keys())[0]]['preds'])} 轮")

print(f"\n{'模型':<15} {'WF R²':>8} {'WF MAE':>10} {'WF MAPE':>8} {'训练R²':>8} {'过拟合程度':>10}")
print("-" * 65)

wf_summary = {}
for name in models_trained:
    preds = np.array(wf_results[name]['preds'])
    actuals = np.array(wf_results[name]['actuals'])
    wf_r2 = r2_score(actuals, preds)
    wf_mae = mean_absolute_error(actuals, preds)
    wf_mape = mean_absolute_percentage_error(actuals, preds) * 100
    train_r2 = results_train.get(name, {}).get('r2', np.nan)
    overfit = train_r2 - wf_r2 if not np.isnan(train_r2) else np.nan
    print(f"{name:<15} {wf_r2:>8.4f} {wf_mae:>10.2f} {wf_mape:>8.2f}% {train_r2:>8.4f} {overfit:>10.4f}")
    wf_summary[name] = {'wf_r2': wf_r2, 'wf_mae': wf_mae, 'wf_mape': wf_mape,
                         'train_r2': train_r2, 'overfit': overfit,
                         'preds': preds, 'actuals': actuals,
                         'dates': wf_results[name]['dates']}

print("-" * 65)

# 选择 Walk-Forward 表现最好的模型
best_wf_name = max(wf_summary, key=lambda k: wf_summary[k]['wf_r2'])
best_wf = wf_summary[best_wf_name]
print(f"\n✅ Walk-Forward 最佳模型: {best_wf_name}  (WF R²={best_wf['wf_r2']:.4f})")

# ============================================================
#  PART 7：Walk-Forward 可视化
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# 图1：Walk-Forward 滚动 R²
ax = axes[0]
window_roll = 20
for name in models_trained:
    preds = np.array(wf_results[name]['preds'])
    actuals = np.array(wf_results[name]['actuals'])
    roll_r2 = pd.Series([r2_score(actuals[max(0,i-window_roll):i+1],
                                   preds[max(0,i-window_roll):i+1])
                         if i >= window_roll else np.nan
                         for i in range(len(preds))])
    ax.plot(wf_results[name]['dates'][window_roll:], roll_r2.values[window_roll:],
            color=colors[name], alpha=.8, lw=1.5, label=f"{name} (WF R²={wf_summary[name]['wf_r2']:.4f})")

ax.axhline(0, color='black', lw=1, ls='--')
ax.set_ylabel('Walk-Forward R2 (Rolling 20)', fontsize=12)
ax.set_title('Walk-Forward Rolling R2 -- Real Generalization Over Time', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=.3)
ax.set_ylim(-1, 1)

# 图2：Walk-Forward 预测 vs 实际（所有模型叠加）
ax2 = axes[1]
wf_dates = np.array(wf_results[list(models_trained.keys())[0]]['dates'])
actuals_arr = np.array(wf_results[list(models_trained.keys())[0]]['actuals'])

ax2.plot(wf_dates, actuals_arr, 'k-', lw=2, alpha=.9, label='Actual Price', zorder=10)

for name in models_trained:
    preds = np.array(wf_results[name]['preds'])
    ax2.plot(wf_dates, preds, color=colors[name], alpha=.65, lw=1.2, label=f"{name}")

ax2.set_ylabel('Price (CNY/ton)', fontsize=12)
ax2.set_title('Walk-Forward Weekly Forecast vs Actual (Window=80w)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=9, loc='upper left', ncol=5)
ax2.grid(True, alpha=.3)

# 图3：各模型 Walk-Forward R² 柱状图对比
ax3 = axes[2]
wf_r2_vals = [wf_summary[n]['wf_r2'] for n in models_trained]
wf_mape_vals = [wf_summary[n]['wf_mape'] for n in models_trained]
train_r2_vals = [wf_summary[n]['train_r2'] for n in models_trained]
overfit_vals = [wf_summary[n]['overfit'] for n in models_trained]

x = np.arange(len(models_trained))
width = 0.25

bars1 = ax3.bar(x - width, train_r2_vals, width, label='Train R2', color='#90CAF9', edgecolor='black', lw=.5)
bars2 = ax3.bar(x, wf_r2_vals, width, label='Walk-Forward R²', color='#1565C0', edgecolor='black', lw=.5)
bars3 = ax3.bar(x + width, overfit_vals, width, label='Overfitting', color='#FF5722', edgecolor='black', lw=.5)

ax3.set_xticks(x)
ax3.set_xticklabels(list(models_trained.keys()), fontsize=12)
ax3.set_ylabel('R2 / Overfitting', fontsize=12)
ax3.set_title('Train R2 vs Walk-Forward R2 vs Overfitting', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.axhline(0, color='black', lw=1)
ax3.grid(True, alpha=.3, axis='y')

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=8)

ax3.set_ylim(-0.2, 1.1)

plt.tight_layout()
plt.savefig('v2_09_walkforward_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Walk-Forward 分析图已保存: v2_09_walkforward_analysis.png")

# ============================================================
#  PART 8：全量训练最佳模型预测
# ============================================================
print(f"\n使用 {best_wf_name} 做未来6周预测...")

best_model = models_trained[best_wf_name]

last_date = df_v2['date'].iloc[-1]
last_price = df_v2['close'].iloc[-1]
n_pred = 6
future_dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(n_pred)]
future_yw = [f"{d.isocalendar()[0]}_{str(d.isocalendar()[1]).zfill(2)}" for d in future_dates]

buffer_len = 30
buf = df_v2.iloc[-buffer_len:].copy().reset_index(drop=True)
predictions = []

for step in range(n_pred):
    b = buf.copy()
    n = len(b)
    feat = {}
    feat['ret_1w'] = (b['close'].iloc[-1] - b['close'].iloc[-2]) / b['close'].iloc[-2] if n >= 2 else 0
    feat['ret_4w'] = (b['close'].iloc[-1] - b['close'].iloc[-5]) / b['close'].iloc[-5] if n >= 5 else feat['ret_1w']
    feat['ret_8w'] = (b['close'].iloc[-1] - b['close'].iloc[-9]) / b['close'].iloc[-9] if n >= 9 else feat['ret_4w']
    feat['mom_diff'] = feat['ret_1w'] - feat['ret_4w']
    ma5 = b['close'].iloc[-5:].mean() if n >= 5 else b['close'].iloc[-1]
    ma20 = b['close'].iloc[-20:].mean() if n >= 20 else b['close'].mean()
    feat['bias_5'] = (b['close'].iloc[-1] - ma5) / ma5
    feat['bias_20'] = (b['close'].iloc[-1] - ma20) / ma20
    rets = b['close'].pct_change().dropna()
    feat['vol_4w'] = rets.iloc[-4:].std() if len(rets) >= 4 else rets.std()
    feat['vol_8w'] = rets.iloc[-8:].std() if len(rets) >= 8 else rets.std()
    feat['range_pct'] = (b['high'].iloc[-1] - b['low'].iloc[-1]) / b['close'].iloc[-1]
    vol_ma5 = b['volume'].iloc[-5:].mean() if n >= 5 else b['volume'].iloc[-1]
    feat['vol_ratio'] = b['volume'].iloc[-1] / vol_ma5 if vol_ma5 > 0 else 1
    feat['oi_chg'] = (b['oi'].iloc[-1] - b['oi'].iloc[-2]) / b['oi'].iloc[-2] if n >= 2 and b['oi'].iloc[-2] != 0 else 0
    feat['oi_chg_4w'] = (b['oi'].iloc[-1] - b['oi'].iloc[-5]) / b['oi'].iloc[-5] if n >= 5 and b['oi'].iloc[-5] != 0 else 0
    if has_wheat:
        wp = b['wheat_profit'].dropna()
        if len(wp) >= 3:
            sl, ic = np.polyfit(range(len(wp.iloc[-8:])), wp.iloc[-8:].values, 1)
            feat['wheat_profit'] = sl * len(wp.iloc[-8:]) + ic
        else:
            feat['wheat_profit'] = b['wheat_profit'].iloc[-1]
        feat['wheat_profit_ma4'] = b['wheat_profit'].iloc[-4:].mean() if n >= 4 else b['wheat_profit'].iloc[-1]
    if 'settle_spread' in all_features:
        feat['settle_spread'] = b['settle'].iloc[-1] - b['close'].iloc[-1] if pd.notna(b['settle'].iloc[-1]) else 0
    feat_vec = np.array([feat[c] for c in all_features]).reshape(1, -1)
    feat_scaled = scaler.transform(feat_vec)
    pred_price = best_model.predict(feat_scaled)[0]
    predictions.append(pred_price)

    new_row = b.iloc[-1:].copy()
    new_row['date'] = future_dates[step]
    new_row['close'] = pred_price
    new_row['open'] = pred_price
    new_row['high'] = pred_price * (1 + abs(feat['range_pct']) * 0.5)
    new_row['low'] = pred_price * (1 - abs(feat['range_pct']) * 0.5)
    new_row['settle'] = pred_price
    new_row['year_week'] = future_yw[step]
    if has_wheat and 'wheat_profit' in new_row.columns:
        new_row['wheat_profit'] = feat.get('wheat_profit', new_row['wheat_profit'].iloc[0])
    buf = pd.concat([buf, new_row], ignore_index=True)

predictions = np.array(predictions)

# 用 Walk-Forward 的 MAE 作为不确定性估计
wf_mae_val = best_wf['wf_mae']
spread = np.linspace(0.4, 1.2, n_pred) * wf_mae_val
upper_1s = predictions + spread
lower_1s = predictions - spread
upper_2s = predictions + 2 * spread
lower_2s = predictions - 2 * spread

wk_chg = np.zeros(n_pred)
wk_chg[0] = predictions[0] - last_price
for i in range(1, n_pred):
    wk_chg[i] = predictions[i] - predictions[i - 1]

cum_chg = predictions - last_price
cum_chg_pct = cum_chg / last_price * 100

def arrow(val):
    if val > 15: return 'Surge'
    elif val > 5: return 'Rise'
    elif val > -5: return 'Flat'
    elif val > -15: return 'Fall'
    else: return 'Crash'

print(f"\n基准: {last_price:.2f} 元/吨 ({last_date.strftime('%Y-%m-%d')})")
print(f"预测: {future_yw[0]} ~ {future_yw[-1]}")
print(f"\n{'周次':<10} {'日期':<12} {'预测价':>8} {'周涨跌':>8} {'累计涨跌':>9} {'趋势':>6}")
print("-" * 60)
for i in range(n_pred):
    print(f"{future_yw[i]:<10} {future_dates[i].strftime('%Y-%m-%d'):<12} "
          f"{predictions[i]:>8.2f} {wk_chg[i]:>+8.2f} {cum_chg[i]:>+9.2f} {arrow(wk_chg[i]):>6}")

# ============================================================
#  PART 9：预测图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]

n_hist = min(80, len(df_v2))
hd = df_v2['date'].iloc[-n_hist:].values
hp = df_v2['close'].iloc[-n_hist:].values

ax.plot(hd, hp, 'k-', lw=2, label='Historical Close')
ax.plot(future_dates, predictions, color='#4CAF50', lw=2.5, marker='D', ms=7,
        label=f'{best_wf_name} 预测 (WF R²={best_wf["wf_r2"]:.4f})')

ax.fill_between(future_dates, lower_1s, upper_1s, alpha=.25, color='#4CAF50', label='±1σ')
ax.fill_between(future_dates, lower_2s, upper_2s, alpha=.10, color='#4CAF50', label='±2σ')
ax.axvline(x=last_date, color='red', ls='--', lw=1.5, alpha=.5, label='Forecast Start')

for d, p in zip(future_dates, predictions):
    ax.annotate(f'{p:.0f}', xy=(d, p), xytext=(0, 16),
                textcoords='offset points', ha='center',
                fontsize=10, fontweight='bold', color='#4CAF50',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1))

ax.set_title(f'{best_wf_name} Forecast | WF R2={best_wf["wf_r2"]:.4f} WF MAE=+-{wf_mae_val:.0f} ({future_yw[0]}~{future_yw[-1]})',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Price (CNY/ton)', fontsize=12)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=.3)

ax2 = axes[1]
colors_c = ['#4CAF50' if c > 0 else '#F44336' for c in cum_chg]
ax2.bar(range(n_pred), cum_chg, color=colors_c, edgecolor='black', lw=.5, alpha=.8)
ax2.set_xticks(range(n_pred))
ax2.set_xticklabels([f'{yw}\n{d.strftime("%m-%d")}' for yw, d in zip(future_yw, future_dates)])
ax2.axhline(0, color='black', lw=.8)
for i, c in enumerate(cum_chg):
    ax2.annotate(f'{c:+.1f}\n({cum_chg_pct[i]:+.1f}%)', xy=(i, c),
                 xytext=(0, 10 if c > 0 else -18), textcoords='offset points',
                 ha='center', fontsize=9, fontweight='bold')
ax2.set_ylabel('Change vs Baseline (CNY)', fontsize=12)
ax2.set_title(f'Weekly Forecast vs Baseline {last_price:.2f} CNY/ton', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=.3, axis='y')

plt.tight_layout()
plt.savefig('v2_10_bestmodel_forecast_wf.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 预测图已保存: v2_10_bestmodel_forecast_wf.png")

# ============================================================
#  PART 10：最终汇总表
# ============================================================
print("\n" + "=" * 75)
print("  最终模型对比汇总")
print("=" * 75)
print(f"\n{'模型':<15} {'训练R²':>8} {'WF R²':>8} {'WF MAE':>10} {'WF MAPE':>8} {'过拟合':>8}")
print("-" * 65)
for name in models_trained:
    s = wf_summary[name]
    print(f"{name:<15} {s['train_r2']:>8.4f} {s['wf_r2']:>8.4f} {s['wf_mae']:>10.2f} {s['wf_mape']:>8.2f}% {s['overfit']:>8.4f}")
print("-" * 65)
print(f"\n✅ 推荐模型: {best_wf_name}  (WF R²={best_wf['wf_r2']:.4f}, 过拟合={best_wf['overfit']:.4f})")
print(f"   置信区间基于 WF MAE=±{wf_mae_val:.0f} 元/吨 构建")
print("=" * 75)

print("\n生成图表:")
print("  - v2_09_walkforward_analysis.png   Walk-Forward 验证分析")
print("  - v2_10_bestmodel_forecast_wf.png 最佳模型预测（基于WF评估）")
