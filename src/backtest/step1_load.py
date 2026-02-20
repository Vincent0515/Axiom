from pathlib import Path
import pandas as pd

feature_files = list((Path("data") / "features").glob("*_feat.parquet"))
if not feature_files:
    raise FileNotFoundError("No feature files found in data/features")

path = feature_files[0]
df = pd.read_parquet(path)

print("Using file:", path)
print("Columns:", list(df.columns))
#print(df[["Date", "Close", "ret_1d", "ma_20", "vol_20"]].head(25))

df = df.sort_values("Date")

# Strategy signal: long when Close > MA20
df["signal"] = (df["Close"] > df["ma_20"]).astype(int)

#print(df[["Date", "Close", "ma_20", "signal"]].head(30))

# Market daily return is already in ret_1d
# Strategy return uses yesterday's signal to avoid look-ahead bias
df["strategy_ret"] = df["signal"].shift(1) * df["ret_1d"]

#print(df[["Date", "ret_1d", "signal", "strategy_ret"]].head(35))

#net_profit
df["equity"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
print(df[["Date", "equity"]].head(35))
print("Total return:", df["equity"].iloc[-1] - 1)

#Markdown Calculation
rolling_max = df["equity"].cummax()
drawdown = df["equity"] / rolling_max - 1
max_dd = drawdown.min()

print("Max Drawdown:", max_dd)