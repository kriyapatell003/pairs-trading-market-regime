# pairs-trading-market-regime
Quant project testing pairs trading with market conditions
# 📊 Pairs Trading with Market Regime Detection

## 🧠 Overview
This project explores a quantitative trading strategy based on pairs trading and analyzes how market conditions impact its performance.

---

## 🔗 Strategy

- Selected correlated stocks (e.g., HDFC Bank & ICICI Bank)
- Calculated spread using hedge ratio (beta)
- Used Z-score to detect divergence
- Entered trades when spread deviates significantly
- Exited when spread normalized

---

## 📈 Market Regime

Market classified using NIFTY 50:

- Bull → Price above 200-day moving average  
- Bear → Price below 200-day moving average  

---

## 📊 Key Insights

- Strategy performs better in stable markets  
- Performance drops during strong trends  
- Market conditions significantly affect results  

---

## ⚙️ Tech Stack

- Python  
- pandas  
- numpy  
- matplotlib  
- statsmodels  
- yfinance  

---

## 🚀 Future Improvements

- Add transaction costs  
- Improve regime detection  
- Test multiple stock pairs  
- Optimize strategy parameters  

---

## 📁 Files

- `pairs_trading_strategy.py` → main code  
- `results.csv` → output data  

---

## 💡 Note

This is a learning project focused on understanding data-driven trading strategies.
