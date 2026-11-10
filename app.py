
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="GANJ AI PRO",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background:#080d16;color:#e8edf7}
h1,h2,h3 {color:#e9bd45}
[data-testid="stMetric"] {
 background:#111a29;padding:15px;border-radius:12px;
 border:1px solid #26364d
}
</style>
""", unsafe_allow_html=True)

BASE = "https://data-api.binance.vision"
HORIZON = 10

@st.cache_data(ttl=60)
def get_candles(symbol, interval, limit=1000):
    url = BASE + "/api/v3/klines"
    r = requests.get(
        url,
        params={"symbol":symbol,"interval":interval,"limit":limit},
        timeout=20
    )
    r.raise_for_status()
    raw = r.json()
    cols = ["time","open","high","low","close","volume",
            "close_time","quote_volume","trades","taker_buy",
            "taker_quote","ignore"]
    df = pd.DataFrame(raw,columns=cols)
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c],errors="coerce")
    df["time"] = pd.to_datetime(df["time"],unit="ms",utc=True)
    now = int(time.time()*1000)
    df = df[df["close_time"].astype("int64") < now]
    return df.reset_index(drop=True)

def rsi(s,n=14):
    d=s.diff()
    up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def adx(df,n=14):
    h,l,c=df.high,df.low,df.close
    up=h.diff()
    down=-l.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0),index=df.index)
    minus=pd.Series(np.where((down>up)&(down>0),down,0),index=df.index)
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/n,adjust=False).mean()
    p=100*plus.ewm(alpha=1/n,adjust=False).mean()/atr
    m=100*minus.ewm(alpha=1/n,adjust=False).mean()/atr
    dx=100*(p-m).abs()/(p+m).replace(0,np.nan)
    return dx.ewm(alpha=1/n,adjust=False).mean()

FEATURES=[
    "ret1","ret2","ret3","ret6","rsi14",
    "macd_pct","macd_signal_pct","ema20_gap",
    "ema50_gap","ema200_gap","atr_pct","adx14",
    "vol_ratio","range_pct","body_pct"
]

def make_features(df):
    x=pd.DataFrame(index=df.index)
    c,h,l,o,v=df.close,df.high,df.low,df.open,df.volume
    x["ret1"]=c.pct_change()
    x["ret2"]=c.pct_change(2)
    x["ret3"]=c.pct_change(3)
    x["ret6"]=c.pct_change(6)
    x["rsi14"]=rsi(c)/100
    e12=c.ewm(span=12,adjust=False).mean()
    e26=c.ewm(span=26,adjust=False).mean()
    macd=e12-e26
    sig=macd.ewm(span=9,adjust=False).mean()
    x["macd_pct"]=macd/c
    x["macd_signal_pct"]=sig/c
    for n in [20,50,200]:
        x[f"ema{n}_gap"]=c/c.ewm(span=n,adjust=False).mean()-1
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False).mean()
    x["atr_pct"]=atr/c
    x["adx14"]=adx(df)/100
    x["vol_ratio"]=v/v.rolling(20).mean()
    x["range_pct"]=(h-l)/c
    x["body_pct"]=(c-o)/o
    return x.replace([np.inf,-np.inf],np.nan),atr

def trend(df):
    c=df.close
    e20=c.ewm(span=20,adjust=False).mean().iloc[-1]
    e50=c.ewm(span=50,adjust=False).mean().iloc[-1]
    e200=c.ewm(span=200,adjust=False).mean().iloc[-1]
    if c.iloc[-1]>e20>e50>e200:
        return "صعودی",1
    if c.iloc[-1]<e20<e50<e200:
        return "نزولی",-1
    return "خنثی",0

def levels(df):
    recent=df.tail(240).reset_index(drop=True)
    highs,lows=[],[]
    for i in range(2,len(recent)-2):
        if recent.high.iloc[i]>=recent.high.iloc[i-2:i+3].max():
            highs.append(float(recent.high.iloc[i]))
        if recent.low.iloc[i]<=recent.low.iloc[i-2:i+3].min():
            lows.append(float(recent.low.iloc[i]))
    price=float(df.close.iloc[-1])
    supports=sorted([v for v in lows if v<price],reverse=True)[:4]
    resistances=sorted([v for v in highs if v>price])[:4]
    return supports,resistances

def forecast_model(df):
    feats,atr_series=make_features(df)
    data=feats.copy()
    for h in range(1,HORIZON+1):
        data[f"y{h}"]=np.log(df.close.shift(-h)/df.close)
    data=data.dropna()
    if len(data)<350:
        raise ValueError("تاریخچه معتبر کافی نیست.")
    x=data[FEATURES]
    y=data[[f"y{i}" for i in range(1,HORIZON+1)]]
    split=int(len(data)*0.8)
    train_end=split-HORIZON
    if train_end<200 or len(data)-split<30:
        raise ValueError("نمونه کافی برای آزمون مدل وجود ندارد.")
    model=RandomForestRegressor(
        n_estimators=160,
        max_depth=7,
        min_samples_leaf=4,
        max_features=0.8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(x.iloc[:train_end],y.iloc[:train_end])
    test_pred=model.predict(x.iloc[split:])
    test_true=y.iloc[split:].values
    accuracy=float(
        np.mean(np.sign(test_true[:,0])==np.sign(test_pred[:,0]))*100
    )
    mae=float(mean_absolute_error(test_true[:,0],test_pred[:,0]))
    model.fit(x,y)
    latest=feats[FEATURES].dropna().iloc[[-1]]
    predictions=model.predict(latest)[0]
    price=float(df.close.iloc[-1])
    atr=float(atr_series.iloc[-1])
    result=[]
    prev=price
    for i,lr in enumerate(predictions,1):
        close=max(price*0.000001,price*np.exp(np.clip(lr,-0.20,0.20)))
        pad=max(atr*0.45,price*0.001)
        result.append({
            "کندل":i,
            "Open":prev,
            "High":max(prev,close)+pad,
            "Low":max(0.00000001,min(prev,close)-pad),
            "Close":close,
            "تغییر از قیمت مبنا (%)":(close/price-1)*100
        })
        prev=close
    return result,accuracy,mae,len(data),feats,atr

st.title("🟡 GANJ AI PRO")
st.caption("موتور تحلیل و پیش‌بینی ارزهای دیجیتال | مرجع اصلی 4H")

c1,c2=st.columns([3,1])
with c1:
    symbol=st.text_input("نماد بایننس", "BTCUSDT").upper().strip()
with c2:
    st.write("")
    st.write("")
    run=st.button("🚀 تحلیل کامل",use_container_width=True)

if run:
    if not symbol.endswith("USDT") or not symbol[:-4].isalnum():
        st.error("نماد معتبر مانند BTCUSDT وارد کنید.")
        st.stop()

    try:
        with st.spinner("دریافت داده زنده و تحلیل بازار..."):
            df=get_candles(symbol,"4h",1000)
            daily=get_candles(symbol,"1d",260)
            hourly=get_candles(symbol,"1h",260)

            if len(df)<400 or len(daily)<200 or len(hourly)<200:
                raise ValueError("تاریخچه این ارز کافی نیست.")

            predicted,accuracy,mae,samples,features,atr=forecast_model(df)
            t4,s4=trend(df)
            td,sd=trend(daily)
            th,sh=trend(hourly)
            price=float(df.close.iloc[-1])
            last=features.iloc[-1]
            supports,resistances=levels(df)

            score=s4*2+sd+sh
            direction="LONG" if score>=2 else "SHORT" if score<=-2 else "WAIT"
            risk=1.5*atr
            sl=price-risk if direction=="LONG" else price+risk
            tp=price+2*(price-sl) if direction=="LONG" else price-2*(sl-price)

        st.success("تحلیل با موفقیت انجام شد")

        st.header(f"{symbol} | قیمت: {price:,.8f}")
        a,b,c=st.columns(3)
        a.metric("روند 4H",t4)
        b.metric("روند روزانه",td)
        c.metric("روند 1H",th)

        st.subheader("شاخص‌های تکنیکال")
        a,b,c,d=st.columns(4)
        a.metric("RSI",f"{last.rsi14*100:.2f}")
        b.metric("MACD","مثبت" if last.macd_pct>last.macd_signal_pct else "منفی")
        c.metric("ADX",f"{last.adx14*100:.2f}")
        d.metric("ATR %",f"{atr/price*100:.2f}%")

        st.subheader("نمودار تاریخی 4H")
        hist=df.tail(180).set_index("time")
        st.line_chart(hist["close"],use_container_width=True)

        st.subheader("🔮 پیش‌بینی ۱۰ کندل آینده")
        future=pd.DataFrame(predicted)
        st.line_chart(future.set_index("کندل")["Close"],use_container_width=True)
        st.dataframe(future,use_container_width=True,hide_index=True)

        st.subheader("حمایت و مقاومت")
        a,b=st.columns(2)
        with a:
            st.write("حمایت‌ها")
            for v in supports:
                st.write(f"🟢 {v:,.8f}")
        with b:
            st.write("مقاومت‌ها")
            for v in resistances:
                st.write(f"🔴 {v:,.8f}")

        st.subheader("سیگنال تحلیلی")
        a,b,c=st.columns(3)
        a.metric("جهت",direction)
        b.metric("حد ضرر",f"{sl:,.8f}")
        c.metric("هدف 2R",f"{tp:,.8f}")

        st.subheader("ارزیابی مدل")
        a,b,c=st.columns(3)
        a.metric("نمونه‌های آموزش نهایی",samples)
        b.metric("دقت جهت آزمون گذشته",f"{accuracy:.2f}%")
        c.metric("MAE بازده لگاریتمی",f"{mae:.6f}")

        st.caption(
            "این خروجی پیش‌بینی آماری است، نه قیمت قطعی آینده یا دستور خرید و فروش. "
            "دقت آزمون گذشته تضمین عملکرد آینده نیست."
        )
        st.caption("آخرین به‌روزرسانی: "+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    except Exception as e:
        st.error(f"خطا در تحلیل: {e}")

else:
    st.info("نماد را وارد کن و روی تحلیل کامل بزن.")
