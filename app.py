from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from functools import lru_cache
from datetime import datetime, timezone
import time
import os

app = FastAPI(title="GANJ AI PRO", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE = os.getenv("BINANCE_BASE_URL", "https://data-api.binance.vision")
INTERVAL = "4h"
HORIZON = 10

def get_json(path, params=None):
    try:
        r = requests.get(BASE + path, params=params, timeout=18)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"خطا در دریافت داده از Binance: {e}")

def candles(symbol, interval="4h", limit=1000):
    raw = get_json("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    cols = ["time","open","high","low","close","volume","close_time","quote_volume","trades","taker_buy","taker_quote","ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    # آخرین کندل ممکن است هنوز باز باشد؛ برای تحلیل و آموزش فقط کندل‌های بسته
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") < now_ms].copy()
    return df.reset_index(drop=True)

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def adx(df, n=14):
    h, l, c = df.high, df.low, df.close
    up, down = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / atr
    dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

FEATURES = ["ret1","ret2","ret3","ret6","rsi14","macd_pct","macd_signal_pct",
            "ema20_gap","ema50_gap","ema200_gap","atr_pct","adx14","vol_ratio",
            "range_pct","body_pct"]

def make_features(df):
    x = pd.DataFrame(index=df.index)
    c, h, l, o, v = df.close, df.high, df.low, df.open, df.volume
    x["ret1"] = c.pct_change()
    x["ret2"] = c.pct_change(2)
    x["ret3"] = c.pct_change(3)
    x["ret6"] = c.pct_change(6)
    x["rsi14"] = rsi(c, 14) / 100
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    x["macd_pct"] = macd / c
    x["macd_signal_pct"] = sig / c
    for n in [20, 50, 200]:
        x[f"ema{n}_gap"] = c / c.ewm(span=n, adjust=False).mean() - 1
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    x["atr_pct"] = atr / c
    x["adx14"] = adx(df) / 100
    x["vol_ratio"] = v / v.rolling(20).mean()
    x["range_pct"] = (h-l) / c
    x["body_pct"] = (c-o) / o
    return x.replace([np.inf, -np.inf], np.nan), atr

def trend_summary(df):
    c = df.close
    e20 = c.ewm(span=20, adjust=False).mean().iloc[-1]
    e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
    e200 = c.ewm(span=200, adjust=False).mean().iloc[-1]
    if c.iloc[-1] > e20 > e50 > e200: return "صعودی", 1
    if c.iloc[-1] < e20 < e50 < e200: return "نزولی", -1
    return "خنثی / ترکیبی", 0

def levels(df):
    recent = df.tail(240).reset_index(drop=True)
    highs, lows = [], []
    for i in range(2, len(recent)-2):
        if recent.high.iloc[i] >= recent.high.iloc[i-2:i+3].max(): highs.append(float(recent.high.iloc[i]))
        if recent.low.iloc[i] <= recent.low.iloc[i-2:i+3].min(): lows.append(float(recent.low.iloc[i]))
    price = float(df.close.iloc[-1])
    supports = sorted([z for z in lows if z < price], reverse=True)[:4]
    resistances = sorted([z for z in highs if z > price])[:4]
    return supports, resistances

def model_forecast(df):
    feats, atr_series = make_features(df)
    data = feats.copy()
    for h in range(1, HORIZON+1):
        data[f"y{h}"] = np.log(df.close.shift(-h) / df.close)
    data = data.dropna()
    if len(data) < 350:
        raise HTTPException(status_code=422, detail="تاریخچه کافی نیست؛ حداقل حدود ۳۵۰ نمونه معتبر لازم است.")
    x = data[FEATURES]
    ycols = [f"y{i}" for i in range(1, HORIZON+1)]
    y = data[ycols]
    split = int(len(data)*0.80)
    if split < 200 or len(data)-split < 30:
        raise HTTPException(status_code=422, detail="نمونه کافی برای آموزش و آزمون مدل وجود ندارد.")
    # آزمون زمانی: آموزش فقط روی گذشته و ارزیابی روی بخش بعدی
    model = RandomForestRegressor(n_estimators=160, max_depth=7, min_samples_leaf=4,
                                  max_features=0.8, n_jobs=-1, random_state=42)
    # Purge the final HORIZON training rows so their future labels cannot
    # overlap the beginning of the test period.
    train_end = max(1, split - HORIZON)
    model.fit(x.iloc[:train_end], y.iloc[:train_end])
    test_pred = model.predict(x.iloc[split:])
    test_true = y.iloc[split:].values
    mae = float(mean_absolute_error(test_true[:,0], test_pred[:,0]))
    direction_acc = float(np.mean(np.sign(test_true[:,0]) == np.sign(test_pred[:,0])) * 100)
    # مدل نهایی پس از گزارش آزمون، روی تمام داده‌های دارای برچسب آموزش می‌بیند
    model.fit(x, y)
    latest_x = feats[FEATURES].dropna().iloc[[-1]]
    predicted_returns = model.predict(latest_x)[0]
    price = float(df.close.iloc[-1])
    atr = float(atr_series.iloc[-1])
    out = []
    prev = price
    for i, lr in enumerate(predicted_returns, start=1):
        close = max(price * 0.000001, price * float(np.exp(np.clip(lr, -0.20, 0.20))))
        # دامنه کندل تخمینی است و از ATR تاریخی مقیاس می‌گیرد؛ قیمت واقعی آینده نیست.
        pad = max(atr * 0.45, price * 0.001)
        high = max(prev, close) + pad
        low = max(0.00000001, min(prev, close) - pad)
        out.append({"step": i, "open": prev, "high": high, "low": low, "close": close,
                    "change_pct": (close/price-1)*100})
        prev = close
    return out, mae, direction_acc, float(atr/price*100)

def analyze(symbol):
    if not symbol.endswith("USDT") or not symbol.replace("USDT","").isalnum():
        raise HTTPException(status_code=400, detail="فعلاً فقط جفت‌ارزهای اسپات USDT پشتیبانی می‌شوند.")
    df = candles(symbol, "4h", 1000)
    if len(df) < 400:
        raise HTTPException(status_code=422, detail="داده تاریخی کافی برای این نماد وجود ندارد.")
    forecast, mae, acc, atr_pct = model_forecast(df)
    d1 = candles(symbol, "1d", 260)
    h1 = candles(symbol, "1h", 260)
    t4, s4 = trend_summary(df)
    td, sd = trend_summary(d1)
    th, sh = trend_summary(h1)
    supports, resistances = levels(df)
    f, _ = make_features(df)
    last = f.iloc[-1]
    rsi_value = float(last.rsi14*100)
    macd_state = "مثبت" if last.macd_pct > last.macd_signal_pct else "منفی"
    # امتیاز توصیفی، نه دستور معامله
    score = s4*2 + sd + sh
    direction = "LONG" if score >= 2 else ("SHORT" if score <= -2 else "WAIT")
    price = float(df.close.iloc[-1])
    sl = price - 1.5*(atr_pct/100*price) if direction=="LONG" else price + 1.5*(atr_pct/100*price)
    tp = price + 2*(price-sl) if direction=="LONG" else price - 2*(sl-price)
    hist = [{"time": int(row.time.timestamp()), "open": float(row.open), "high": float(row.high),
             "low": float(row.low), "close": float(row.close)} for row in df.tail(180).itertuples()]
    return {
        "symbol":symbol, "price":price, "updated":datetime.now(timezone.utc).isoformat(),
        "history":hist, "forecast":forecast,
        "trends":{"4h":t4,"1d":td,"1h":th},
        "indicators":{"rsi":rsi_value,"macd":macd_state,"adx":float(last.adx14*100),
                      "atr_pct":atr_pct,"ema20_gap_pct":float(last.ema20_gap*100)},
        "levels":{"supports":supports,"resistances":resistances},
        "signal":{"side":direction,"stop_loss":sl,"take_profit_2r":tp,"score":score},
        "model":{"name":"RandomForest multi-horizon","train_samples":int(len(data_count(df))),
                 "test_direction_accuracy_pct":acc,"test_mae_log_return":mae,
                 "warning":"دقت آزمون گذشته تضمین عملکرد آینده نیست."}
    }

def data_count(df):
    f,_=make_features(df)
    return f.dropna()

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

@app.get("/api/symbols")
def symbols():
    info = get_json("/api/v3/exchangeInfo")
    syms = sorted([s["symbol"] for s in info["symbols"]
                   if s.get("status")=="TRADING" and s.get("quoteAsset")=="USDT"
                   and s.get("isSpotTradingAllowed", True)])
    return {"symbols":syms}

@app.get("/api/analyze/{symbol}")
def api_analyze(symbol: str):
    try:
        return analyze(symbol.upper())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطای تحلیل: {str(e)}")
