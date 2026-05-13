"""
玉米期货价格预测 — 多模型对比版
对比 ElasticNet / Ridge / RandomForest / XGBoost

运行方式: python corn_compare.py
依赖: pandas numpy matplotlib scikit-learn xgboost
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
matplotlib.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'SimHei', 'Songti SC', 'Arial Unicode MS']
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

# ============================================================
#  PART 3：标准化
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
#  PART 4：训练四个模型
# ============================================================
print("\n" + "=" * 70)
print("  训练四个模型...")
print("=" * 70)

# --- ElasticNet ---
model_en_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                            n_alphas=200, cv=10, max_iter=50000, random_state=42, n_jobs=-1)
model_en_cv.fit(X_scaled, y)
model_en = ElasticNet(alpha=model_en_cv.alpha_, l1_ratio=model_en_cv.l1_ratio_,
                       max_iter=50000, random_state=42)
model_en.fit(X_scaled, y)
y_pred_en = model_en.predict(X_scaled)

# --- Ridge ---
model_ridge = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=10)
model_ridge.fit(X_scaled, y)
y_pred_ridge = model_ridge.predict(X_scaled)

# --- RandomForest ---
model_rf = RandomForestRegressor(
    n_estimators=200, max_depth=6, min_samples_leaf=3,
    max_features=0.7, random_state=42, n_jobs=-1
)
model_rf.fit(X_scaled, y)
y_pred_rf = model_rf.predict(X_scaled)

# --- XGBoost ---
try:
    from xgboost import XGBRegressor
    model_xgb = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbosity=0
    )
    model_xgb.fit(X_scaled, y)
    y_pred_xgb = model_xgb.predict(X_scaled)
    xgb_ok = True
except ImportError:
    print("⚠️ XGBoost 未安装，跳过")
    xgb_ok = False

# ============================================================
#  PART 5：评估汇总
# ============================================================
models = {
    'ElasticNet': (model_en, y_pred_en),
    'Ridge': (model_ridge, y_pred_ridge),
    'RandomForest': (model_rf, y_pred_rf),
}
if xgb_ok:
    models['XGBoost'] = (model_xgb, y_pred_xgb)

print(f"\n{'模型':<15} {'R²':>8} {'MAE':>10} {'MAPE':>8} {'残差σ':>10}")
print("-" * 55)

results = {}
for name, (model, yp) in models.items():
    r2 = r2_score(y, yp)
    mae = mean_absolute_error(y, yp)
    mape = mean_absolute_percentage_error(y, yp) * 100
    resid_std = np.std(y - yp)
    print(f"{name:<15} {r2:>8.4f} {mae:>10.2f} {mape:>8.2f}% {resid_std:>10.2f}")
    results[name] = {'r2': r2, 'mae': mae, 'mape': mape, 'resid_std': resid_std, 'y_pred': yp}

print("-" * 55)

# 选择最佳模型（R² 最高）
best_name = max(results, key=lambda k: results[k]['r2'])
best_res = results[best_name]
print(f"\n✅ 最佳模型: {best_name}  (R²={best_res['r2']:.4f})")

# ============================================================
#  PART 6：拟合对比图
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]

colors = {'ElasticNet': '#E91E63', 'Ridge': '#FF9800', 'RandomForest': '#2196F3', 'XGBoost': '#4CAF50'}
dates = df_v2['date'].values

ax.plot(dates, y, 'k-', lw=2.5, label='Actual Price', zorder=10)

for name, res in results.items():
    ax.plot(dates, res['y_pred'], color=colors[name], lw=1.5, alpha=.75,
            label=f"{name} (R²={res['r2']:.4f})")

ax.set_title(f'Model Comparison | Samples={len(y)} Features={len(all_features)}',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Price (CNY/ton)', fontsize=12)
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=.3)

ax2 = axes[1]
width = 0.2
model_names = list(results.keys())
r2_vals = [results[n]['r2'] for n in model_names]
bars = ax2.bar(range(len(model_names)), r2_vals,
               color=[colors[n] for n in model_names], edgecolor='black', lw=.5, alpha=.8)
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, fontsize=12)
ax2.set_ylabel('R2', fontsize=12)
ax2.set_title('R2 Comparison', fontsize=12, fontweight='bold')
ax2.axhline(best_res['r2'], color='red', ls='--', lw=1, alpha=.5)
for bar, val in zip(bars, r2_vals):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_ylim(0, max(r2_vals) * 1.15)
ax2.grid(True, alpha=.3, axis='y')

plt.tight_layout()
plt.savefig('v2_04_model_compare_fit.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 对比拟合图已保存: v2_04_model_compare_fit.png")

# ============================================================
#  PART 7：特征重要性对比
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for idx, (name, (model, _)) in enumerate(models.items()):
    ax = axes[idx]
    if hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
        color_list = ['#E91E63' if c > 0 else '#4CAF50' for c in model.coef_]
        label = 'Coefficient'
    else:
        imp = model.feature_importances_
        color_list = '#2196F3'
        label = 'Importance'

    imp_df = pd.Series(imp, index=all_features).sort_values(ascending=True)
    imp_df.plot.barh(ax=ax, color=color_list, edgecolor='black', lw=.3)
    ax.set_xlabel(label, fontsize=10)
    ax.set_title(f'{name}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=.3, axis='x')

plt.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('v2_05_feature_importance_compare.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 特征重要性对比图已保存: v2_05_feature_importance_compare.png")

# ============================================================
#  PART 8：未来6周滚动预测（用最佳模型）
# ============================================================
print(f"\n使用 {best_name} 做未来6周预测...")

best_model = models[best_name][0]

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

resid_std = best_res['resid_std']
spread = np.linspace(0.4, 1.2, n_pred) * resid_std
upper_1s = predictions + spread
lower_1s = predictions - spread
upper_2s = predictions + 2 * spread
lower_2s = predictions - 2 * spread

wk_chg = np.zeros(n_pred)
wk_chg[0] = predictions[0] - last_price
for i in range(1, n_pred):
    wk_chg[i] = predictions[i] - predictions[i - 1]

wk_chg_pct = np.zeros(n_pred)
wk_chg_pct[0] = wk_chg[0] / last_price * 100
for i in range(1, n_pred):
    wk_chg_pct[i] = wk_chg[i] / predictions[i - 1] * 100

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
#  PART 9：预测图（从实际值出发，无拟合线）
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(18, 11), gridspec_kw={'height_ratios': [3, 1]})
ax = axes[0]

n_hist = min(80, len(df_v2))
hd = df_v2['date'].iloc[-n_hist:].values
hp = df_v2['close'].iloc[-n_hist:].values

ax.plot(hd, hp, 'k-', lw=2, label='Historical Close')

# 预测线从 last_price 出发（不是拟合线末端）
ax.plot(future_dates, predictions, color='#4CAF50', lw=2.5, marker='D', ms=7,
        label=f'{best_name} 预测')

ax.fill_between(future_dates, lower_1s, upper_1s, alpha=.25, color='#4CAF50', label='+-1sigma')
ax.fill_between(future_dates, lower_2s, upper_2s, alpha=.10, color='#4CAF50', label='+-2sigma')
ax.axvline(x=last_date, color='red', ls='--', lw=1.5, alpha=.5, label='Forecast Start')

for d, p in zip(future_dates, predictions):
    ax.annotate(f'{p:.0f}', xy=(d, p), xytext=(0, 16),
                textcoords='offset points', ha='center',
                fontsize=10, fontweight='bold', color='#4CAF50',
                arrowprops=dict(arrowstyle='->', color='#4CAF50', lw=1))

ax.set_title(f'{best_name} Forecast | R2={best_res["r2"]:.4f} MAPE={best_res["mape"]:.2f}% ' 
             f'sigma=+-{resid_std:.0f} ({future_yw[0]}~{future_yw[-1]})',
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
plt.savefig('v2_06_bestmodel_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ 最佳模型预测图已保存: v2_06_bestmodel_forecast.png")

# ============================================================
#  PART 10：四个模型预测对比
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(hd, hp, 'k-', lw=2.5, label='Historical Close', zorder=10)

for name, (model, _) in models.items():
    preds_all = []
    buf2 = df_v2.iloc[-buffer_len:].copy().reset_index(drop=True)
    for step in range(n_pred):
        b = buf2.copy()
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
        pred_price = model.predict(feat_scaled)[0]
        preds_all.append(pred_price)
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
        buf2 = pd.concat([buf2, new_row], ignore_index=True)

    ax.plot(future_dates, preds_all, color=colors[name], lw=2, marker='o', ms=5, alpha=.85,
            label=f"{name} (R²={results[name]['r2']:.4f})")

ax.axvline(x=last_date, color='red', ls='--', lw=1.5, alpha=.5)
ax.set_title(f'4-Model Forecast Comparison | Baseline={last_price:.2f} ({last_date.strftime("%Y-%m-%d")})',
             fontsize=14, fontweight='bold')
ax.set_ylabel('Price (CNY/ton)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig('v2_07_allmodel_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 四模型预测对比图已保存: v2_07_allmodel_forecast.png")

# ============================================================
#  PART 11：残差分析
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
for name, res in results.items():
    resid = y - res['y_pred']
    ax.hist(resid, bins=25, alpha=.5, label=name, color=colors[name], edgecolor='black', lw=.3)
ax.axvline(0, color='black', lw=1)
ax.set_xlabel('Residual (CNY)', fontsize=12)
ax.set_title('Residual Distribution', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=.3)

ax2 = axes[1]
for name, res in results.items():
    resid = y - res['y_pred']
    ax2.plot(dates, resid, color=colors[name], alpha=.7, lw=1.2, label=name)
ax2.axhline(0, color='black', lw=1)
ax2.axhline(resid_std, color='red', ls='--', lw=.8, alpha=.5)
ax2.axhline(-resid_std, color='red', ls='--', lw=.8, alpha=.5)
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Residual (CNY)', fontsize=12)
ax2.set_title('Residual Time Series', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=.3)

plt.tight_layout()
plt.savefig('v2_08_residuals.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ 残差分析图已保存: v2_08_residuals.png")

print("\n" + "=" * 70)
print("  完成！生成图表：")
print("  - v2_04_model_compare_fit.png    多模型拟合对比")
print("  - v2_05_feature_importance_compare.png  特征重要性对比")
print("  - v2_06_bestmodel_forecast.png    最佳模型预测")
print("  - v2_07_allmodel_forecast.png     四模型预测对比")
print("  - v2_08_residuals.png             残差分析")
print("=" * 70)
