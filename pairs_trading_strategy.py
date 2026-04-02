"""
Pairs Trading Strategy with Market Regime Detection

Author: KRIYA 

Description:
This project implements a quantitative pairs trading strategy using HDFCBANK and ICICIBANK.
It includes:
- Cointegration testing
- Z-score based signals
- Market regime detection (Bull/Bear)
- Backtesting with transaction costs
- Performance metrics and visualization

For educational purposes only.
"""
import warnings; warnings.filterwarnings('ignore')
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller

─────────────────────────────────────────────────────────────────

1. DOWNLOAD DATA

─────────────────────────────────────────────────────────────────

tickers = ['HDFCBANK.NS', 'ICICIBANK.NS', '^NSEI']
raw = yf.download(tickers, start='2019-01-01', end='2024-01-01', auto_adjust=True)
prices = raw['Close'].dropna()
prices.columns = ['HDFCBANK', 'ICICIBANK', 'NIFTY']
print(f"✅ Downloaded {len(prices)} rows")

─────────────────────────────────────────────────────────────────

2. COINTEGRATION TEST (Engle-Granger)

─────────────────────────────────────────────────────────────────

score, pval, crits = coint(prices['HDFCBANK'], prices['ICICIBANK'])
print(f"\n📊 Cointegration p-value: {pval:.4f}")
print("✅ Cointegrated" if pval < 0.05 else "❌ Not cointegrated")

─────────────────────────────────────────────────────────────────

3. HEDGE RATIO VIA OLS

─────────────────────────────────────────────────────────────────

X = prices[['ICICIBANK']].values
y = prices['HDFCBANK'].values
model = LinearRegression().fit(X, y)
beta  = model.coef_[0]
print(f"\n📐 Hedge Ratio (β): {beta:.4f}")

─────────────────────────────────────────────────────────────────

4. SPREAD + Z-SCORE

─────────────────────────────────────────────────────────────────

WINDOW = 60
prices['SPREAD']      = prices['HDFCBANK'] - beta * prices['ICICIBANK']
prices['SPREAD_MEAN'] = prices['SPREAD'].rolling(WINDOW).mean()
prices['SPREAD_STD']  = prices['SPREAD'].rolling(WINDOW).std()
prices['ZSCORE']      = (prices['SPREAD'] - prices['SPREAD_MEAN']) / prices['SPREAD_STD']
prices = prices.dropna()

─────────────────────────────────────────────────────────────────

5. GENERATE POSITIONS (no look-ahead bias)

─────────────────────────────────────────────────────────────────

ENTRY = 2.0
EXIT  = 0.5
STOP  = 3.5   # Stop-loss Z level

position, positions = 0, []
for _, row in prices.iterrows():
z = row['ZSCORE']
if position == 0:
if   z < -ENTRY: position =  1
elif z >  ENTRY: position = -1
elif position ==  1:
if abs(z) < EXIT or z < -STOP: position = 0
elif position == -1:
if abs(z) < EXIT or z >  STOP: position = 0
positions.append(position)

prices['POSITION'] = pd.Series(positions, index=prices.index).shift(1).fillna(0)

─────────────────────────────────────────────────────────────────

6. MARKET REGIME (Nifty 200 DMA)

─────────────────────────────────────────────────────────────────

prices['NIFTY_MA200'] = prices['NIFTY'].rolling(200).mean()
prices['REGIME']     = (prices['NIFTY'] > prices['NIFTY_MA200']).astype(int)

─────────────────────────────────────────────────────────────────

7. BACKTEST WITH TRANSACTION COSTS

─────────────────────────────────────────────────────────────────

TC_RATE = 0.0005   # 5 bps per trade side

prices['RET_HDFC']  = prices['HDFCBANK'].pct_change()
prices['RET_ICICI'] = prices['ICICIBANK'].pct_change()
prices['STRAT_RET'] = prices['POSITION'] * (prices['RET_HDFC'] - beta * prices['RET_ICICI'])

trade_days = prices['POSITION'].diff().abs() > 0
prices['STRAT_RET'] -= trade_days * TC_RATE * 2
prices['CUMRET']     = (1 + prices['STRAT_RET']).cumprod() - 1

─────────────────────────────────────────────────────────────────

8. PERFORMANCE SUMMARY

─────────────────────────────────────────────────────────────────

print("\n📈 PERFORMANCE SUMMARY")
print("-" * 45)
for regime, label in [(1, 'Bull'), (0, 'Bear'), (None, 'All ')]:
subset = prices['STRAT_RET'] if regime is None else \
prices[prices['REGIME'] == regime]['STRAT_RET']
ann_r = subset.mean() * 252
ann_v = subset.std()  * np.sqrt(252)
sharpe = ann_r / ann_v if ann_v > 0 else 0
maxdd  = ((1 + subset).cumprod() / (1 + subset).cumprod().cummax() - 1).min()
print(f"{label} | Ret={ann_r:+.1%} | Vol={ann_v:.1%} | Sharpe={sharpe:.2f} | MaxDD={maxdd:.1%}")

─────────────────────────────────────────────────────────────────

9. VISUALISE

─────────────────────────────────────────────────────────────────

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.patch.set_facecolor('#09090f')
for ax in [ax1, ax2, ax3]:
ax.set_facecolor('#111118')
ax.tick_params(colors='#7a7a96')
for spine in ax.spines.values(): spine.set_edgecolor('#222230')
ax.grid(True, alpha=0.1, color='#444')

ax1.plot(prices.index, prices['SPREAD'], color='#7c6af7', lw=1)
ax1.set_title('Spread (HDFCBANK − β·ICICIBANK)', color='#e8e8f0')
ax1.set_ylabel('₹', color='#7a7a96')

ax2.plot(prices.index, prices['ZSCORE'], color='#f7a04a', lw=1)
ax2.axhline( 2,  color='#f25c5c', ls='--', lw=1)
ax2.axhline(-2,  color='#3ecfb2', ls='--', lw=1)
ax2.axhline( 0,  color='#ffffff', ls='-',  lw=0.5, alpha=0.2)
ax2.fill_between(prices.index, prices['ZSCORE'], 0,
where=prices['ZSCORE']>2,  alpha=0.2, color='#f25c5c')
ax2.fill_between(prices.index, prices['ZSCORE'], 0,
where=prices['ZSCORE']<-2, alpha=0.2, color='#3ecfb2')
ax2.set_title('Z-Score | Red=Short Spread | Green=Long Spread', color='#e8e8f0')
ax2.set_ylabel('Z-Score', color='#7a7a96')

ax3.fill_between(prices.index, 0, 1,
where=prices['REGIME']==1, alpha=0.07, color='#3ecfb2',
transform=ax3.get_xaxis_transform(), label='Bull')
ax3.fill_between(prices.index, 0, 1,
where=prices['REGIME']==0, alpha=0.07, color='#f25c5c',
transform=ax3.get_xaxis_transform(), label='Bear')
ax3.plot(prices.index, prices['CUMRET'] * 100, color='#7c6af7', lw=1.5)
ax3.axhline(0, color='white', lw=0.5, alpha=0.2)
ax3.set_title('Cumulative Returns (%) by Market Regime', color='#e8e8f0')
ax3.set_ylabel('Return (%)', color='#7a7a96')
ax3.legend(facecolor='#16161f', labelcolor='#c0c0d8')

fig.suptitle(
'Pairs Trading: HDFCBANK vs ICICIBANK | 2019–2024',
color='#e8e8f0', fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('pairs_trading_result.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Chart saved: pairs_trading_result.png")
