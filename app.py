import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timezone

st.set_page_config(page_title="GANJ AI PRO", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:#080d16;color:#e8edf7}
[data-testid="stMetric"]{background:#111a29;border:1px solid #26364d;padding:12px;border-radius:12px}
h1,h2,h3{color:#e9bd45}
div[data-testid="stDataFrame"]{border:1px solid #26364d;border-radius:10px}
</style>
""", unsafe_allow_html=True)

BASE = "https://data-api.binance.vision"
MASTER = "4h"
HORIZON = 10
TIMEOUT = 15

@st.cache_data(ttl=45, show_spinner=False)
def api_get(path, params=None):
    r = requests.get(BASE + path, params=params, timeout=TIMEOUT,
                     headers={"User-Agent":"GANJ-AI-PRO/1.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=45, show_spinner=False)
def get_klines(symbol, interval, limit=1000):
    raw = api_get("/api/v3/klines", {"symbol":symbol, "interval":interval, "limit":limit})
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    d = pd.DataFrame(raw, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume","taker_buy_base","taker_buy_quote"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(d["open_time"], unit="ms", utc=True)
    now_ms = int(datetime.now(timezone.utc).timestamp()*1000)
    # Exclude the currently forming candle: all analysis uses closed candles.
    d = d[d["close_time"].astype("int64") < now_ms].copy()
    return d.reset_index(drop=True)

@st.cache_data(ttl=10, show_spinner=False)
def get_book(symbol, limit=100):
    return api_get("/api/v3/depth", {"symbol":symbol, "limit":limit})

@st.cache_data(ttl=10, show_spinner=False)
def get_trades(symbol, limit=500):
    return api_get("/api/v3/aggTrades", {"symbol":symbol, "limit":limit})

@st.cache_data(ttl=30, show_spinner=False)
def get_ticker(symbol):
    return api_get("/api/v3/ticker/24hr", {"symbol":symbol})

def indicators(df):
    d=df.copy()
    c,h,l,o,v=d.close,d.high,d.low,d.open,d.volume
    d["ema20"]=c.ewm(span=20,adjust=False).mean()
    d["ema50"]=c.ewm(span=50,adjust=False).mean()
    d["ema200"]=c.ewm(span=200,adjust=False).mean()
    delta=c.diff()
    gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
    loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["rsi"]=100-100/(1+gain/loss.replace(0,np.nan))
    e12=c.ewm(span=12,adjust=False).mean()
    e26=c.ewm(span=26,adjust=False).mean()
    d["macd"]=e12-e26
    d["macd_signal"]=d.macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"]=tr.ewm(alpha=1/14,adjust=False).mean()
    up=h.diff()
    down=-l.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0),index=d.index).ewm(alpha=1/14,adjust=False).mean()
    minus=pd.Series(np.where((down>up)&(down>0),down,0),index=d.index).ewm(alpha=1/14,adjust=False).mean()
    pdi=100*plus/d.atr.replace(0,np.nan)
    mdi=100*minus/d.atr.replace(0,np.nan)
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["adx"]=dx.ewm(alpha=1/14,adjust=False).mean()
    d["vol_ma20"]=v.rolling(20).mean()
    d["vol_ratio"]=v/d.vol_ma20
    d["buy_ratio"]=d.taker_buy_base/d.volume.replace(0,np.nan)
    return d

def trend(d):
    z=d.iloc[-1]
    if z.close>z.ema20>z.ema50>z.ema200: return "صعودی",1
    if z.close<z.ema20<z.ema50<z.ema200: return "نزولی",-1
    return "ترکیبی",0

def sr_levels(d, n=300):
    q=d.tail(n).reset_index(drop=True)
    hs=[]; ls=[]
    for i in range(2,len(q)-2):
        if q.high.iloc[i] >= q.high.iloc[i-2:i+3].max(): hs.append(float(q.high.iloc[i]))
        if q.low.iloc[i] <= q.low.iloc[i-2:i+3].min(): ls.append(float(q.low.iloc[i]))
    p=float(d.close.iloc[-1])
    return sorted([x for x in ls if x<p],reverse=True)[:5], sorted([x for x in hs if x>p])[:5]

def structure(d):
    q=d.tail(100).reset_index(drop=True)
    ph=[]; pl=[]
    for i in range(2,len(q)-2):
        if q.high.iloc[i]>=q.high.iloc[i-2:i+3].max(): ph.append((i,float(q.high.iloc[i])))
        if q.low.iloc[i]<=q.low.iloc[i-2:i+3].min(): pl.append((i,float(q.low.iloc[i])))
    if len(ph)>=2 and len(pl)>=2:
        hh=ph[-1][1]>ph[-2][1]; hl=pl[-1][1]>pl[-2][1]
        lh=ph[-1][1]<ph[-2][1]; ll=pl[-1][1]<pl[-2][1]
        if hh and hl: return "HH / HL — ساختار صعودی"
        if lh and ll: return "LH / LL — ساختار نزولی"
    return "ساختار نامشخص / رنج"

def divergence(d):
    q=d.tail(80).reset_index(drop=True)
    piv=[]
    for i in range(2,len(q)-2):
        if q.low.iloc[i]<=q.low.iloc[i-2:i+3].min(): piv.append(("L",i))
        if q.high.iloc[i]>=q.high.iloc[i-2:i+3].max(): piv.append(("H",i))
    lows=[i for t,i in piv if t=="L"]
    highs=[i for t,i in piv if t=="H"]
    out=[]
    if len(lows)>=2:
        a,b=lows[-2:]
        if q.low.iloc[b]<q.low.iloc[a] and q.rsi.iloc[b]>q.rsi.iloc[a]: out.append("واگرایی مثبت RSI")
    if len(highs)>=2:
        a,b=highs[-2:]
        if q.high.iloc[b]>q.high.iloc[a] and q.rsi.iloc[b]<q.rsi.iloc[a]: out.append("واگرایی منفی RSI")
    return out or ["واگرایی واضح شناسایی نشد"]

def fvg_zones(d):
    q=d.tail(120).reset_index(drop=True)
    zones=[]
    for i in range(2,len(q)):
        if q.low.iloc[i] > q.high.iloc[i-2]:
            zones.append({"نوع":"FVG صعودی","پایین":float(q.high.iloc[i-2]),"بالا":float(q.low.iloc[i])})
        if q.high.iloc[i] < q.low.iloc[i-2]:
            zones.append({"نوع":"FVG نزولی","پایین":float(q.high.iloc[i]),"بالا":float(q.low.iloc[i-2])})
    return zones[-8:]

def orderflow(book, trades, d):
    bids=book.get("bids",[])[:50]; asks=book.get("asks",[])[:50]
    bq=sum(float(x[0])*float(x[1]) for x in bids)
    aq=sum(float(x[0])*float(x[1]) for x in asks)
    imb=(bq-aq)/(bq+aq) if bq+aq else 0
    buy=sum(float(t["q"]) for t in trades if not t["m"])
    sell=sum(float(t["q"]) for t in trades if t["m"])
    total=buy+sell
    taker=(buy-sell)/total if total else 0
    spread=0
    if bids and asks:
        bp=float(bids[0][0]); ap=float(asks[0][0])
        spread=(ap-bp)/((ap+bp)/2)*10000
    candle_buy=float(d.iloc[-1].buy_ratio) if pd.notna(d.iloc[-1].buy_ratio) else np.nan
    return {"bid_quote":bq,"ask_quote":aq,"imbalance":imb,"taker_buy":buy,
            "taker_sell":sell,"taker_imbalance":taker,"spread_bps":spread,
            "candle_buy_ratio":candle_buy}

def analog_forecast(d, horizon=10, lookback=900, neighbors=12):
    # Similarity on normalized, lagged price/volume/indicator context.
    z=indicators(d).dropna().reset_index(drop=True)
    if len(z)<260: raise ValueError("برای پیش‌بینی، تاریخچه معتبر کافی نیست.")
    cols=["ret1","ret2","ret4","rsi_n","ema_gap","vol_n","range_n"]
    z["ret1"]=z.close.pct_change()
    z["ret2"]=z.close.pct_change(2)
    z["ret4"]=z.close.pct_change(4)
    z["rsi_n"]=z.rsi/100
    z["ema_gap"]=z.close/z.ema50-1
    z["vol_n"]=z.vol_ratio.clip(0,5)/5
    z["range_n"]=(z.high-z.low)/z.close
    z=z.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
    # Current feature vector uses last 6 closed bars.
    def vec(i):
        return z.loc[i-5:i,cols].to_numpy().flatten()
    end=len(z)-horizon-1
    candidates=range(max(5,end-lookback),end)
    target=vec(len(z)-1)
    dist=[]
    for i in candidates:
        # Keep analogue future fully before the latest bar to avoid leakage.
        if i+horizon >= len(z)-1: continue
        v=vec(i)
        scale=np.nanstd(np.vstack([v,target]),axis=0)
        scale=np.where(scale<1e-8,1,scale)
        dist.append((float(np.mean(((v-target)/scale)**2)),i))
    if len(dist)<20: raise ValueError("نمونه تاریخی مشابه کافی پیدا نشد.")
    chosen=sorted(dist)[:neighbors]
    paths=[]
    for _,i in chosen:
        base=float(z.close.iloc[i])
        paths.append((z.close.iloc[i+1:i+1+horizon].to_numpy()/base).tolist())
    med=np.median(np.array(paths),axis=0)
    price=float(d.close.iloc[-1]); prev=price
    atr=float(z.atr.iloc[-1])
    out=[]
    for k,ratio in enumerate(med,1):
        close=price*float(ratio)
        pad=max(atr*0.45,price*0.001)
        out.append({"کندل":k,"Open":prev,"High":max(prev,close)+pad,
                    "Low":max(0.00000001,min(prev,close)-pad),"Close":close})
        prev=close
    agreement=float(np.mean([np.sign(p[-1]-1)==np.sign(med[-1]-1) for p in paths])*100)
    return out,len(chosen),agreement,paths

def chart_figure(hist, future):
    fig=go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.time,open=hist.open,high=hist.high,low=hist.low,close=hist.close,
        name="کندل‌های بسته‌شده 4H",
        increasing_line_color="#16c79a",decreasing_line_color="#ff647c"
    ))
    last_time=hist.time.iloc[-1]
    future_x=[last_time+pd.Timedelta(hours=4*i) for i in range(1,len(future)+1)]
    f=pd.DataFrame(future)
    fig.add_trace(go.Candlestick(
        x=future_x,open=f["Open"],high=f["High"],low=f["Low"],close=f["Close"],
        name="۱۰ کندل تخمینی",
        increasing_line_color="#e9bd45",decreasing_line_color="#ff9d56"
    ))
    fig.update_layout(
        template="plotly_dark",height=570,xaxis_rangeslider_visible=False,
        paper_bgcolor="#080d16",plot_bgcolor="#080d16",
        margin=dict(l=10,r=10,t=35,b=10),legend=dict(orientation="h")
    )
    return fig

st.title("🟡 GANJ AI PRO")
st.caption("تحلیل چندلایه بازار کریپتو | مرجع ثابت 4H | پیش‌بینی ۱۰ کندل آینده")
symbol=st.text_input("نماد اسپات بایننس را وارد کن", "BTCUSDT").strip().upper()
col1,col2=st.columns([1,4])
with col1:
    run=st.button("🔍 تحلیل کامل",type="primary",use_container_width=True)
with col2:
    st.caption("داده عمومی بایننس: کندل‌ها، دفتر سفارش، معاملات تجمیعی و آمار ۲۴ ساعته")

if run:
    if not symbol.endswith("USDT") or not symbol[:-4].isalnum():
        st.error("نماد معتبر مانند BTCUSDT وارد کن.")
        st.stop()
    try:
        with st.spinner("در حال دریافت داده‌های بازار و اجرای تحلیل..."):
            h4=indicators(get_klines(symbol,"4h",1000))
            d1=indicators(get_klines(symbol,"1d",300))
            h1=indicators(get_klines(symbol,"1h",500))
            btc=indicators(get_klines("BTCUSDT","4h",500)) if symbol!="BTCUSDT" else h4
            book=get_book(symbol,100)
            trades=get_trades(symbol,500)
            ticker=get_ticker(symbol)
            if min(len(h4),len(d1),len(h1))<220:
                raise ValueError("تاریخچه کافی برای تحلیل چندتایم‌فریمی موجود نیست.")
            future,n_analogs,agreement,paths=analog_forecast(h4,HORIZON)
            flow=orderflow(book,trades,h4)
            t4,s4=trend(h4); td,sd=trend(d1); th,sh=trend(h1); tb,sb=trend(btc)
            price=float(h4.close.iloc[-1]); atr=float(h4.atr.iloc[-1])
            supports,resistances=sr_levels(h4)
            divs=divergence(h4); fvg=fvg_zones(h4)
            structure_text=structure(h4)
            bull=0; bear=0
            bull += 2 if s4>0 else 0; bear += 2 if s4<0 else 0
            bull += 1 if sd>0 else 0; bear += 1 if sd<0 else 0
            bull += 1 if sh>0 else 0; bear += 1 if sh<0 else 0
            bull += 1 if sb>0 else 0; bear += 1 if sb<0 else 0
            bull += 1 if flow["imbalance"]>0.08 else 0
            bear += 1 if flow["imbalance"]<-0.08 else 0
            bull += 1 if flow["taker_imbalance"]>0.08 else 0
            bear += 1 if flow["taker_imbalance"]<-0.08 else 0
            bull += 1 if h4.iloc[-1].rsi>52 and h4.iloc[-1].macd>h4.iloc[-1].macd_signal else 0
            bear += 1 if h4.iloc[-1].rsi<48 and h4.iloc[-1].macd<h4.iloc[-1].macd_signal else 0
            side="LONG" if bull>=bear+2 else "SHORT" if bear>=bull+2 else "NO TRADE"
            sl=price-1.8*atr if side=="LONG" else price+1.8*atr
            tp1=price+(atr if side=="LONG" else -atr)
            tp2=price+(2*atr if side=="LONG" else -2*atr)
            tp3=price+(3*atr if side=="LONG" else -3*atr)
            hist=h4.tail(160).copy()
        st.success("تحلیل انجام شد — آخرین کندل‌های بسته‌شده استفاده شده‌اند.")
        st.subheader(f"{symbol} | {price:,.8f}")
        ch=st.columns(4)
        ch[0].metric("تغییر ۲۴ ساعته",f"{float(ticker['priceChangePercent']):.2f}%")
        ch[1].metric("حجم ۲۴ ساعته",f"{float(ticker['quoteVolume']):,.0f} USDT")
        ch[2].metric("بیشترین قیمت ۲۴h",f"{float(ticker['highPrice']):,.8f}")
        ch[3].metric("کمترین قیمت ۲۴h",f"{float(ticker['lowPrice']):,.8f}")
        st.plotly_chart(chart_figure(hist,future),use_container_width=True)
        st.caption("کندل‌های زرد/نارنجی، برآورد آماری بر پایه مسیرهای تاریخی مشابه‌اند؛ قیمت واقعی آینده نیستند.")
        st.subheader("لوحة بازار و روند چندتایم‌فریمی")
        c=st.columns(4)
        c[0].metric("روند 4H",t4); c[1].metric("روند روزانه",td)
        c[2].metric("روند 1H",th); c[3].metric("فیلتر BTC 4H",tb)
        c=st.columns(5)
        c[0].metric("RSI 4H",f"{h4.iloc[-1].rsi:.1f}")
        c[1].metric("ADX 4H",f"{h4.iloc[-1].adx:.1f}")
        c[2].metric("MACD", "مثبت" if h4.iloc[-1].macd>h4.iloc[-1].macd_signal else "منفی")
        c[3].metric("EMA20",f"{h4.iloc[-1].ema20:,.6g}")
        c[4].metric("حجم نسبت به میانگین",f"{h4.iloc[-1].vol_ratio:.2f}x")
        st.subheader("تابلوخوانی و جریان سفارش")
        c=st.columns(4)
        c[0].metric("عدم‌تعادل دفتر سفارش",f"{flow['imbalance']*100:.1f}%")
        c[1].metric("عدم‌تعادل معاملات خریدار/فروشنده",f"{flow['taker_imbalance']*100:.1f}%")
        c[2].metric("فاصله بهترین خرید/فروش",f"{flow['spread_bps']:.2f} bps")
        c[3].metric("نسبت خرید تیکر در کندل",f"{flow['candle_buy_ratio']*100:.1f}%" if pd.notna(flow["candle_buy_ratio"]) else "ناموجود")
        st.caption(f"حجم خرید تیکر در معاملات اخیر: {flow['taker_buy']:.4f} | حجم فروش تیکر: {flow['taker_sell']:.4f}")
        book_rows=[]
        for p,q in book.get("bids",[])[:10]:
            book_rows.append({"سمت":"خرید","قیمت":float(p),"مقدار":float(q),"ارزش USDT":float(p)*float(q)})
        for p,q in book.get("asks",[])[:10]:
            book_rows.append({"سمت":"فروش","قیمت":float(p),"مقدار":float(q),"ارزش USDT":float(p)*float(q)})
        st.dataframe(pd.DataFrame(book_rows),use_container_width=True,hide_index=True)
        st.subheader("ساختار بازار، واگرایی و نواحی")
        st.write("**ساختار:**",structure_text)
        st.write("**واگرایی:**","، ".join(divs))
        st.write("**حمایت‌ها:**"," | ".join(f"{x:,.8f}" for x in supports) or "یافت نشد")
        st.write("**مقاومت‌ها:**"," | ".join(f"{x:,.8f}" for x in resistances) or "یافت نشد")
        st.write("**FVGهای اخیر:**")
        st.dataframe(pd.DataFrame(fvg) if fvg else pd.DataFrame(columns=["نوع","پایین","بالا"]),use_container_width=True,hide_index=True)
        st.subheader("پیش‌بینی و سنجش شباهت تاریخی")
        st.write(f"تعداد مسیرهای تاریخی مشابه: **{n_analogs}** | توافق جهت در همسایه‌ها: **{agreement:.1f}%**")
        st.dataframe(pd.DataFrame(future).style.format("{:,.8f}",subset=["Open","High","Low","Close"]),use_container_width=True,hide_index=True)
        st.subheader("جمع‌بندی سیگنال — صرفاً تحلیلی")
        c=st.columns(5)
        c[0].metric("نتیجه",side)
        c[1].metric("امتیاز صعودی",bull)
        c[2].metric("امتیاز نزولی",bear)
        c[3].metric("حد ضرر تخمینی",f"{sl:,.8g}")
        c[4].metric("هدف 2R تقریبی",f"{tp2:,.8g}")
        st.write(f"اهداف ATR: TP1 `{tp1:,.8g}` | TP2 `{tp2:,.8g}` | TP3 `{tp3:,.8g}`")
        st.info("این ابزار از داده عمومی بازار استفاده می‌کند و سفارش واقعی ثبت نمی‌کند. دفتر سفارش، تصویر لحظه‌ای و قابل تغییر است؛ حجم خرید/فروش معاملات اخیر معادل سفارش‌های قطعی آینده نیست. سیگنال و کندل‌های آینده تخمینی‌اند، نه تضمین سود یا قیمت.")
        st.caption("تحلیل با کندل‌های بسته‌شده انجام شده؛ داده زنده دفتر سفارش و معاملات ممکن است بین دو بار تحلیل تغییر کند.")
        st.caption("زمان گزارش: "+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    except requests.HTTPError as e:
        st.error(f"خطای دریافت داده بایننس: {e}. نماد را بررسی کن یا کمی بعد دوباره تلاش کن.")
    except Exception as e:
        st.error(f"خطا در تحلیل: {e}")
else:
    st.info("نماد دلخواه را وارد کن و «تحلیل کامل» را بزن.")
    st.write("ماژول‌ها: داده تاریخی 4H/1D/1H، دفتر سفارش، معاملات تجمیعی، EMA/RSI/MACD/ADX، ساختار بازار، واگرایی، حمایت/مقاومت، FVG و ۱۰ کندل تخمینی آینده.")
