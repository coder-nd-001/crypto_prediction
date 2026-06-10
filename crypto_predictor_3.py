import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

# =====================================================
#  PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="CryptoSage AI",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
#  CSS
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #07070f; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }

/* ---- NAV TABS ---- */
.nav-bar {
    display: flex; gap: 6px; align-items: center;
    background: #0d0d1a; border: 1px solid #1a1a3e;
    border-radius: 14px; padding: 6px; margin-bottom: 24px;
}
.nav-tab {
    flex: 1; text-align: center; padding: 10px 0;
    border-radius: 10px; font-size: 0.88rem; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
    color: #475569; border: 1px solid transparent;
    letter-spacing: 0.02em;
}
.nav-tab.active {
    background: linear-gradient(135deg,#1e1b4b,#1e1a3e);
    color: #818cf8; border-color: #3730a3;
}
.nav-tab:hover:not(.active) { color: #94a3b8; background: #0f0f1f; }

/* ---- HERO ---- */
.hero-wrap {
    background: linear-gradient(135deg,#0d0d1a 0%,#0a0f2e 50%,#0d0d1a 100%);
    border: 1px solid #1a1a3e; border-radius: 20px;
    padding: 40px 36px 32px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content:''; position:absolute; top:-80px; right:-80px;
    width:300px; height:300px;
    background: radial-gradient(circle,rgba(99,102,241,.15) 0%,transparent 70%);
    border-radius:50%;
}
.hero-badge {
    display:inline-block; background:rgba(99,102,241,.15);
    border:1px solid rgba(99,102,241,.4); color:#818cf8;
    font-size:.72rem; font-weight:600; letter-spacing:.12em;
    text-transform:uppercase; padding:4px 14px; border-radius:100px; margin-bottom:14px;
}
.hero-title {
    font-size:2.6rem; font-weight:700; color:#f1f5f9;
    line-height:1.1; margin:0 0 8px; letter-spacing:-.03em;
}
.hero-title span { background:linear-gradient(90deg,#f59e0b,#fbbf24); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { color:#64748b; font-size:1rem; margin:0; }
.hero-stats { display:flex; gap:28px; margin-top:24px; flex-wrap:wrap; }
.hero-stat { display:flex; flex-direction:column; }
.hero-stat-val { font-family:'JetBrains Mono',monospace; font-size:1.25rem; font-weight:600; color:#f1f5f9; }
.hero-stat-lbl { font-size:.7rem; color:#475569; text-transform:uppercase; letter-spacing:.08em; margin-top:2px; }
.hero-divider  { width:1px; background:#1e293b; align-self:stretch; }

/* ---- KPI CARDS ---- */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }
.kpi-card {
    background:#0f0f1a; border:1px solid #1e1e3a; border-radius:14px;
    padding:18px; position:relative; overflow:hidden;
}
.kpi-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--acc,linear-gradient(90deg,#6366f1,#818cf8)); }
.kpi-label { font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; color:#475569; font-weight:600; margin-bottom:7px; }
.kpi-value { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:600; color:#f1f5f9; line-height:1.1; }
.kpi-ch { font-size:.76rem; margin-top:5px; font-weight:500; }
.kpi-ch.up{color:#34d399;} .kpi-ch.dn{color:#f87171;} .kpi-ch.ne{color:#94a3b8;}

/* ---- SIGNAL BANNER ---- */
.signal-banner { border-radius:14px; padding:18px 22px; display:flex; align-items:center; gap:14px; margin-bottom:20px; border:1px solid; }
.signal-banner.buy  { background:rgba(52,211,153,.08); border-color:rgba(52,211,153,.3); }
.signal-banner.sell { background:rgba(248,113,113,.08); border-color:rgba(248,113,113,.3); }
.signal-banner.hold { background:rgba(251,191,36,.08);  border-color:rgba(251,191,36,.3); }
.sig-icon { font-size:1.9rem; }
.sig-title { font-size:1.15rem; font-weight:700; margin-bottom:2px; }
.sig-title.buy{color:#34d399;} .sig-title.sell{color:#f87171;} .sig-title.hold{color:#fbbf24;}
.sig-desc { font-size:.83rem; color:#64748b; }

/* ---- RISK BADGE ---- */
.risk-row { display:flex; gap:10px; align-items:center; margin-bottom:8px; }
.risk-badge { display:inline-flex; align-items:center; gap:6px; padding:5px 14px; border-radius:100px; font-size:.8rem; font-weight:600; letter-spacing:.05em; }
.risk-low  { background:rgba(52,211,153,.12); color:#34d399; border:1px solid rgba(52,211,153,.3); }
.risk-med  { background:rgba(251,191,36,.12);  color:#fbbf24; border:1px solid rgba(251,191,36,.3); }
.risk-high { background:rgba(248,113,113,.12); color:#f87171; border:1px solid rgba(248,113,113,.3); }

/* ---- MODEL CARDS ---- */
.model-compare { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:20px; }
.model-card { background:#0f0f1a; border:1px solid #1e1e3a; border-radius:12px; padding:16px; text-align:center; }
.model-card.active { border-color:#6366f1; background:rgba(99,102,241,.07); }
.model-name { font-size:.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; margin-bottom:7px; }
.model-pred { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:600; color:#f1f5f9; }
.model-tag  { font-size:.68rem; margin-top:5px; }
.model-tag.at{color:#818cf8;} .model-tag.it{color:#334155;}

/* ---- SECTION HEADER ---- */
.sec-head { display:flex; align-items:center; gap:10px; margin:28px 0 14px; }
.sec-icon { width:30px;height:30px; background:rgba(99,102,241,.15); border:1px solid rgba(99,102,241,.3); border-radius:8px; display:flex;align-items:center;justify-content:center; font-size:.95rem; }
.sec-text { font-size:1.1rem; font-weight:600; color:#e2e8f0; letter-spacing:-.01em; }

/* ---- INSIGHT PILL ---- */
.insight-pill { background:rgba(99,102,241,.07); border:1px solid rgba(99,102,241,.2); border-radius:10px; padding:12px 16px; margin-bottom:9px; font-size:.86rem; color:#94a3b8; display:flex; align-items:flex-start; gap:9px; }
.insight-pill .ic { color:#818cf8; font-size:.95rem; flex-shrink:0; margin-top:1px; }

/* ---- MANUAL MODEL PAGE ---- */
.mm-card { background:#0f0f1a; border:1px solid #1e1e3a; border-radius:16px; padding:24px; margin-bottom:16px; }
.mm-title { font-size:1rem; font-weight:600; color:#e2e8f0; margin-bottom:4px; }
.mm-desc  { font-size:.82rem; color:#475569; margin-bottom:16px; }
.mm-result-box { background:#070710; border:1px solid #1e1e3a; border-radius:12px; padding:20px; text-align:center; }
.mm-result-label { font-size:.72rem; color:#475569; text-transform:uppercase; letter-spacing:.1em; margin-bottom:8px; }
.mm-result-value { font-family:'JetBrains Mono',monospace; font-size:2rem; font-weight:700; color:#f59e0b; }

/* ---- DASH VIEW ---- */
.dash-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:20px; }
.dash-coin-card {
    background:linear-gradient(135deg,#0d0d1a,#0f0f2a);
    border:1px solid #1e1e3a; border-radius:16px; padding:20px;
    position:relative; overflow:hidden; transition:border-color .2s;
}
.dash-coin-card:hover { border-color:#3730a3; }
.dash-coin-top { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px; }
.dash-coin-name { font-size:.78rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
.dash-coin-sym  { font-size:.72rem; color:#334155; }
.dash-coin-price { font-family:'JetBrains Mono',monospace; font-size:1.2rem; font-weight:700; color:#f1f5f9; }
.dash-coin-change { font-size:.8rem; font-weight:600; padding:3px 10px; border-radius:100px; }
.dash-coin-change.up  { background:rgba(52,211,153,.12); color:#34d399; }
.dash-coin-change.dn  { background:rgba(248,113,113,.12); color:#f87171; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] { background:#060610 !important; border-right:1px solid #0f0f2a; }
section[data-testid="stSidebar"] * { color:#94a3b8 !important; }
section[data-testid="stSidebar"] label { color:#64748b !important; font-size:.78rem !important; text-transform:uppercase; letter-spacing:.06em; }

/* ---- FOOTER ---- */
.footer { text-align:center; color:#1e293b; font-size:.76rem; padding:20px 0 6px; border-top:1px solid #0f172a; margin-top:32px; }
.footer span { color:#334155; }
</style>
""", unsafe_allow_html=True)

# =====================================================
#  SESSION STATE — active view
# =====================================================
if "view" not in st.session_state:
    st.session_state.view = "Predictor"

# =====================================================
#  SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("### ₿ CryptoSage AI")
    st.markdown("---")

    coin = st.selectbox(
        "Cryptocurrency",
        ["bitcoin","ethereum","litecoin","solana","cardano","dogecoin"],
        format_func=lambda x: {
            "bitcoin":  "₿ Bitcoin (BTC)",
            "ethereum": "Ξ Ethereum (ETH)",
            "litecoin": "Ł Litecoin (LTC)",
            "solana":   "◎ Solana (SOL)",
            "cardano":  "₳ Cardano (ADA)",
            "dogecoin": "Ð Dogecoin (DOGE)"
        }[x]
    )
    st.markdown("---")
    investment = st.number_input("Investment Amount (₹)", min_value=1000, max_value=10000000, value=10000, step=1000)
    days       = st.slider("Historical Data (Days)", 30, 365, 180)
    st.markdown("---")
    model_choice = st.radio("AI Prediction Model", ["LSTM","Linear Regression","Hybrid (LSTM + LR)"], index=2)
    st.markdown("---")
    show_bollinger   = st.toggle("Bollinger Bands",     value=True)
    show_rsi         = st.toggle("RSI Indicator",       value=True)
    show_macd        = st.toggle("MACD Chart",          value=True)
    show_volume      = st.toggle("Volume Distribution", value=True)
    show_model_cmp   = st.toggle("Model Comparison",    value=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;color:#1e293b;text-align:center'>CryptoSage AI v3.0<br>No API key required for Manual Model</div>", unsafe_allow_html=True)

# =====================================================
#  NAV BAR  (3 view buttons)
# =====================================================
VIEWS = ["📊  Dashboard", "🔮  Predictor", "🧠  Manual Model"]
cols_nav = st.columns(len(VIEWS))
for col, v in zip(cols_nav, VIEWS):
    label = v.split("  ")[1]
    active = "active" if st.session_state.view == label else ""
    if col.button(v, key=f"nav_{label}", use_container_width=True):
        st.session_state.view = label
        st.rerun()

VIEW = st.session_state.view

# =====================================================
#  HELPERS — TECHNICAL INDICATORS
# =====================================================
COIN_SYMBOLS = {"bitcoin":"BTC","ethereum":"ETH","litecoin":"LTC","solana":"SOL","cardano":"ADA","dogecoin":"DOGE"}
symbol = COIN_SYMBOLS[coin]

def compute_rsi(s, period=14):
    d = s.diff(); g = d.clip(lower=0).rolling(period).mean(); l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - (100/(1+g/l.replace(0,np.nan)))

def compute_macd(s, fast=12, slow=26, sig=9):
    ef = s.ewm(span=fast,adjust=False).mean(); es = s.ewm(span=slow,adjust=False).mean()
    m = ef-es; sl = m.ewm(span=sig,adjust=False).mean(); return m, sl, m-sl

def compute_bollinger(s, w=20, n=2):
    sma=s.rolling(w).mean(); std=s.rolling(w).std(); return sma+n*std, sma, sma-n*std

def sec(icon, text):
    st.markdown(f'<div class="sec-head"><div class="sec-icon">{icon}</div><div class="sec-text">{text}</div></div>', unsafe_allow_html=True)

def kpi_html(label, value, change=None, acc="linear-gradient(90deg,#6366f1,#818cf8)"):
    ch = ""
    if change is not None:
        cls = "up" if change>0 else ("dn" if change<0 else "ne")
        sgn = "▲" if change>0 else ("▼" if change<0 else "—")
        ch  = f'<div class="kpi-ch {cls}">{sgn} {abs(change):.2f}%</div>'
    return f'<div class="kpi-card" style="--acc:{acc}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>{ch}</div>'

# =====================================================
#  FETCH LIVE DATA  (cached 5 min)
# =====================================================
@st.cache_data(ttl=300)
def fetch_crypto_data(coin, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
    r   = requests.get(url, params={"vs_currency":"inr","days":days})
    d   = r.json()
    df  = pd.DataFrame(d["prices"], columns=["timestamp","price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    if "total_volumes" in d:
        vdf = pd.DataFrame(d["total_volumes"], columns=["timestamp","volume"])
        vdf["timestamp"] = pd.to_datetime(vdf["timestamp"], unit="ms")
        vdf.set_index("timestamp", inplace=True)
        df = df.join(vdf, how="left")
    if "market_caps" in d:
        mdf = pd.DataFrame(d["market_caps"], columns=["timestamp","market_cap"])
        mdf["timestamp"] = pd.to_datetime(mdf["timestamp"], unit="ms")
        mdf.set_index("timestamp", inplace=True)
        df = df.join(mdf, how="left")
    return df

@st.cache_data(ttl=300)
def fetch_all_coins():
    coins = ["bitcoin","ethereum","litecoin","solana","cardano","dogecoin"]
    rows  = []
    for c in coins:
        try:
            r = requests.get(
                f"https://api.coingecko.com/api/v3/coins/{c}/market_chart",
                params={"vs_currency":"inr","days":2}
            )
            prices = r.json()["prices"]
            p_now  = prices[-1][1];  p_24h = prices[max(0,len(prices)-25)][1]
            chg    = ((p_now-p_24h)/p_24h)*100
            rows.append({"coin":c,"symbol":COIN_SYMBOLS[c],"price":p_now,"change_24h":chg})
        except:
            pass
    return rows

# ===================================================
# ============  VIEW: DASHBOARD  ====================
# ===================================================
if VIEW == "Dashboard":

    st.markdown('<div class="hero-wrap"><div class="hero-badge">Live Market Overview</div><p class="hero-title">₿ <span>CryptoSage</span> Dashboard</p><p class="hero-sub">Real-time snapshot of all tracked cryptocurrencies</p></div>', unsafe_allow_html=True)

    with st.spinner("Loading market snapshot…"):
        all_data = fetch_all_coins()

    if all_data:
        sec("🪙", "Live Prices — All Coins")
        cols3 = st.columns(3)
        for i, row in enumerate(all_data):
            cls   = "up" if row["change_24h"]>=0 else "dn"
            arrow = "▲" if row["change_24h"]>=0 else "▼"
            with cols3[i % 3]:
                st.markdown(f"""
                <div class="dash-coin-card">
                    <div class="dash-coin-top">
                        <div>
                            <div class="dash-coin-name">{row['coin'].capitalize()}</div>
                            <div class="dash-coin-sym">{row['symbol']}/INR</div>
                        </div>
                        <span class="dash-coin-change {cls}">{arrow} {abs(row['change_24h']):.2f}%</span>
                    </div>
                    <div class="dash-coin-price">₹{row['price']:,.2f}</div>
                </div>""", unsafe_allow_html=True)

    # Comparison bar chart
    sec("📊", "24h Price Change Comparison")
    if all_data:
        bar_fig = go.Figure(go.Bar(
            x=[r["symbol"] for r in all_data],
            y=[r["change_24h"] for r in all_data],
            marker_color=["#34d399" if r["change_24h"]>=0 else "#f87171" for r in all_data],
            text=[f"{r['change_24h']:+.2f}%" for r in all_data],
            textposition="outside"
        ))
        bar_fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,20,.6)", height=300,
            font=dict(family="Space Grotesk",color="#64748b",size=12),
            yaxis=dict(title="24h Change %", gridcolor="#0f172a", zeroline=True, zerolinecolor="#1e293b"),
            xaxis=dict(gridcolor="#0f172a"),
            margin=dict(l=10,r=10,t=20,b=10), showlegend=False
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Selected coin mini chart
    sec("📈", f"{symbol} — {days}d Price History")
    with st.spinner("Loading chart…"):
        df_dash = fetch_crypto_data(coin, days)
    dash_fig = go.Figure()
    dash_fig.add_trace(go.Scatter(
        x=df_dash.index, y=df_dash["price"],
        mode="lines", line=dict(color="#f59e0b", width=2.2), fill="tozeroy",
        fillcolor="rgba(245,158,11,.05)", name="Price",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>"
    ))
    dash_fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,20,.6)", height=320,
        font=dict(family="Space Grotesk",color="#64748b",size=11),
        xaxis=dict(gridcolor="#0f172a"), yaxis=dict(gridcolor="#0f172a",tickprefix="₹",tickformat=",.0f"),
        margin=dict(l=10,r=10,t=10,b=10), showlegend=False
    )
    st.plotly_chart(dash_fig, use_container_width=True)

    # KPI row for selected coin
    cp    = df_dash["price"].iloc[-1]
    p1d   = df_dash["price"].iloc[-min(len(df_dash),24)]
    ch1d  = ((cp-p1d)/p1d)*100
    hi    = df_dash["price"].max()
    lo    = df_dash["price"].min()
    vol_d = df_dash["price"].pct_change().std()
    rsi_d = compute_rsi(df_dash["price"]).iloc[-1]
    sec("📋", f"{symbol} Key Stats")
    cards = (
        kpi_html("Current Price",     f"₹{cp:,.2f}",    ch1d,  "linear-gradient(90deg,#6366f1,#818cf8)") +
        kpi_html(f"{days}d High",     f"₹{hi:,.2f}",    None,  "linear-gradient(90deg,#34d399,#6ee7b7)") +
        kpi_html(f"{days}d Low",      f"₹{lo:,.2f}",    None,  "linear-gradient(90deg,#f87171,#fca5a5)") +
        kpi_html("RSI (14)",          f"{rsi_d:.1f}",   None,  "linear-gradient(90deg,#a78bfa,#c4b5fd)") +
        kpi_html("Daily Volatility",  f"{vol_d*100:.2f}%", None,"linear-gradient(90deg,#f59e0b,#fbbf24)")
    )
    st.markdown(f'<div class="kpi-grid">{cards}</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">Built by <span>Nagesh</span> · CryptoSage AI v3.0 · <span>Not financial advice</span></div>', unsafe_allow_html=True)


# ===================================================
# ============  VIEW: PREDICTOR  ====================
# ===================================================
elif VIEW == "Predictor":

    with st.spinner("Fetching live market data…"):
        df = fetch_crypto_data(coin, days)

    current_price = df["price"].iloc[-1]
    p1d_ago       = df["price"].iloc[-min(len(df),24)]
    change_1d     = ((current_price-p1d_ago)/p1d_ago)*100
    p7d_ago       = df["price"].iloc[-min(len(df),200)]
    change_7d     = ((current_price-p7d_ago)/p7d_ago)*100
    high_p        = df["price"].max()
    low_p         = df["price"].min()

    df["rsi"]                            = compute_rsi(df["price"])
    df["ma20"]                           = df["price"].rolling(20).mean()
    df["ma50"]                           = df["price"].rolling(50).mean()
    df["bb_upper"],df["bb_mid"],df["bb_lower"] = compute_bollinger(df["price"])
    df["macd"],df["macd_signal"],df["macd_hist"] = compute_macd(df["price"])

    rsi_current = df["rsi"].iloc[-1]
    rsi_label   = "Overbought" if rsi_current>70 else ("Oversold" if rsi_current<30 else "Neutral")

    mc_str = "—"
    if "market_cap" in df.columns:
        mc = df["market_cap"].iloc[-1]
        mc_str = f"₹{mc/1e12:.2f}T" if mc>=1e12 else f"₹{mc/1e9:.2f}B" if mc>=1e9 else f"₹{mc/1e6:.2f}M"

    chg1d_col = "#34d399" if change_1d>=0 else "#f87171"
    chg1d_arr = "▲" if change_1d>=0 else "▼"

    # HERO
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge">AI Price Predictor · Live</div>
        <p class="hero-title">{symbol} / INR &nbsp;<span>CryptoSage</span></p>
        <p class="hero-sub">Real-time forecasting · Technical analysis · Risk assessment</p>
        <div class="hero-stats">
            <div class="hero-stat"><span class="hero-stat-val">₹{current_price:,.2f}</span><span class="hero-stat-lbl">Current Price</span></div>
            <div class="hero-divider"></div>
            <div class="hero-stat"><span class="hero-stat-val" style="color:{chg1d_col}">{chg1d_arr} {abs(change_1d):.2f}%</span><span class="hero-stat-lbl">24h Change</span></div>
            <div class="hero-divider"></div>
            <div class="hero-stat"><span class="hero-stat-val">{rsi_current:.1f}</span><span class="hero-stat-lbl">RSI · {rsi_label}</span></div>
            <div class="hero-divider"></div>
            <div class="hero-stat"><span class="hero-stat-val">{mc_str}</span><span class="hero-stat-lbl">Market Cap</span></div>
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI CARDS
    cards_html = (
        kpi_html("Current Price",  f"₹{current_price:,.2f}", change_1d, "linear-gradient(90deg,#6366f1,#818cf8)") +
        kpi_html(f"{days}d High",  f"₹{high_p:,.2f}",        None,      "linear-gradient(90deg,#34d399,#6ee7b7)") +
        kpi_html(f"{days}d Low",   f"₹{low_p:,.2f}",         None,      "linear-gradient(90deg,#f87171,#fca5a5)") +
        kpi_html("7d Change",      f"{change_7d:+.2f}%",     change_7d, "linear-gradient(90deg,#f59e0b,#fbbf24)") +
        kpi_html("RSI (14)",       f"{rsi_current:.1f}",     None,      "linear-gradient(90deg,#a78bfa,#c4b5fd)")
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)

    # PRICE CHART
    sec("📈", "Price Chart")
    pfig = go.Figure()
    if show_bollinger:
        pfig.add_trace(go.Scatter(x=df.index,y=df["bb_upper"],mode="lines",line=dict(color="rgba(99,102,241,.3)",width=1,dash="dot"),name="BB Upper"))
        pfig.add_trace(go.Scatter(x=df.index,y=df["bb_lower"],mode="lines",line=dict(color="rgba(99,102,241,.3)",width=1,dash="dot"),fill="tonexty",fillcolor="rgba(99,102,241,.04)",name="BB Band"))
    pfig.add_trace(go.Scatter(x=df.index,y=df["price"],mode="lines",line=dict(color="#f59e0b",width=2.5),name=f"{symbol} Price",hovertemplate="<b>%{x|%d %b %Y}</b><br>₹%{y:,.2f}<extra></extra>"))
    pfig.add_trace(go.Scatter(x=df.index,y=df["ma20"],mode="lines",line=dict(color="#818cf8",width=1.2,dash="dash"),name="MA 20"))
    pfig.add_trace(go.Scatter(x=df.index,y=df["ma50"],mode="lines",line=dict(color="#34d399",width=1.2,dash="dash"),name="MA 50"))
    pfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",height=420,
                       font=dict(family="Space Grotesk",color="#64748b",size=12),
                       legend=dict(orientation="h",yanchor="bottom",y=1.02,x=0,bgcolor="rgba(0,0,0,0)"),
                       xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix="₹",tickformat=",.0f"),
                       hovermode="x unified",margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(pfig, use_container_width=True)

    # RSI + MACD
    if show_rsi or show_macd:
        n = int(show_rsi)+int(show_macd)
        titles = (["RSI (14)"] if show_rsi else []) + (["MACD"] if show_macd else [])
        ifig = make_subplots(rows=n,cols=1,shared_xaxes=True,row_heights=[1]*n,subplot_titles=titles,vertical_spacing=.08)
        r = 1
        if show_rsi:
            ifig.add_trace(go.Scatter(x=df.index,y=df["rsi"],line=dict(color="#a78bfa",width=1.8),name="RSI"),row=r,col=1)
            ifig.add_hline(y=70,line=dict(color="#f87171",dash="dot",width=1),row=r,col=1)
            ifig.add_hline(y=30,line=dict(color="#34d399",dash="dot",width=1),row=r,col=1)
            ifig.add_hrect(y0=70,y1=100,fillcolor="rgba(248,113,113,.04)",line_width=0,row=r,col=1)
            ifig.add_hrect(y0=0, y1=30, fillcolor="rgba(52,211,153,.04)", line_width=0,row=r,col=1)
            r+=1
        if show_macd:
            hc = ["#34d399" if v>=0 else "#f87171" for v in df["macd_hist"].fillna(0)]
            ifig.add_trace(go.Bar(x=df.index,y=df["macd_hist"],marker_color=hc,name="Histogram",opacity=.6),row=r,col=1)
            ifig.add_trace(go.Scatter(x=df.index,y=df["macd"],line=dict(color="#6366f1",width=1.5),name="MACD"),row=r,col=1)
            ifig.add_trace(go.Scatter(x=df.index,y=df["macd_signal"],line=dict(color="#f59e0b",width=1.5,dash="dash"),name="Signal"),row=r,col=1)
        ifig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                           height=200*n,font=dict(family="Space Grotesk",color="#64748b",size=11),
                           legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a"),
                           margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(ifig, use_container_width=True)

    # VOLUME
    if show_volume and "volume" in df.columns and not df["volume"].isna().all():
        sec("📦","Volume Distribution")
        vfig = go.Figure()
        vfig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=df["price"].pct_change().fillna(0).apply(lambda x:"rgba(52,211,153,.5)" if x>=0 else "rgba(248,113,113,.5)"),
            hovertemplate="<b>%{x|%d %b}</b><br>Vol: %{y:,.0f}<extra></extra>"
        ))
        vfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                           height=200,showlegend=False,margin=dict(l=10,r=10,t=10,b=10),
                           xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickformat=".2s"))
        st.plotly_chart(vfig, use_container_width=True)

    # ML PREPROCESSING
    prices_arr = df["price"].values.reshape(-1,1)
    scaler     = MinMaxScaler()
    scaled     = scaler.fit_transform(prices_arr)
    window_size= 15
    X,y_arr    = [],[]
    for i in range(window_size,len(scaled)):
        X.append(scaled[i-window_size:i]); y_arr.append(scaled[i])
    X = np.array(X); y_arr = np.array(y_arr)
    split      = int(len(X)*.8)
    X_tr,X_te  = X[:split],X[split:]
    y_tr,y_te  = y_arr[:split],y_arr[split:]

    lr_model = LinearRegression()
    lr_model.fit(X_tr.reshape(X_tr.shape[0],-1), y_tr)
    lr_p_sc  = lr_model.predict(X_te.reshape(X_te.shape[0],-1)).flatten()
    lr_preds = scaler.inverse_transform(lr_p_sc.reshape(-1,1)).flatten()

    lstm_model = None
    if model_choice in ["LSTM","Hybrid (LSTM + LR)"]:
        with st.spinner("Training LSTM…"):
            lstm_model = Sequential([
                LSTM(64,return_sequences=True,input_shape=(X.shape[1],1)),
                Dropout(.2), LSTM(32), Dropout(.2), Dense(16), Dense(1)
            ])
            lstm_model.compile(optimizer="adam",loss="huber")
            lstm_model.fit(X_tr,y_tr,epochs=8,batch_size=32,verbose=0)
        lstm_p_sc= lstm_model.predict(X_te,verbose=0).flatten()
        lstm_preds= scaler.inverse_transform(lstm_p_sc.reshape(-1,1)).flatten()

    last_w  = scaled[-window_size:]
    X_pred  = np.array([last_w])
    lr_next_sc = lr_model.predict(X_pred.reshape(1,-1))[0][0]
    lr_next = scaler.inverse_transform([[lr_next_sc]])[0][0]

    if lstm_model:
        lstm_next_sc= lstm_model.predict(X_pred,verbose=0)[0][0]
        lstm_next   = scaler.inverse_transform([[lstm_next_sc]])[0][0]
    else:
        lstm_next = lr_next

    if   model_choice=="Linear Regression":  predicted_price = lr_next
    elif model_choice=="LSTM":               predicted_price = lstm_next
    else:                                    predicted_price = (lr_next+lstm_next)/2

    change_percent = ((predicted_price-current_price)/current_price)*100
    y_true         = scaler.inverse_transform(y_te.reshape(-1,1)).flatten()
    lr_mae  = mean_absolute_error(y_true,lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_true,lr_preds))
    lr_mape = np.mean(np.abs((y_true-lr_preds)/y_true))*100
    lr_acc  = max(0,100-lr_mape)
    lstm_acc= None
    if lstm_model:
        lstm_mape = np.mean(np.abs((y_true-lstm_preds)/y_true))*100
        lstm_acc  = max(0,100-lstm_mape)

    # MODEL COMPARISON
    if show_model_cmp:
        sec("🤖","Model Comparison")
        hybrid_next = (lr_next+lstm_next)/2
        is_lr = model_choice=="Linear Regression"
        is_ls = model_choice=="LSTM"
        is_hy = model_choice=="Hybrid (LSTM + LR)"
        st.markdown(f"""
        <div class="model-compare">
            <div class="model-card {'active' if is_lr else ''}">
                <div class="model-name">Linear Regression</div>
                <div class="model-pred">₹{lr_next:,.2f}</div>
                <div class="model-tag {'at' if is_lr else 'it'}">{'✦ Active' if is_lr else f'Acc {lr_acc:.1f}%'}</div>
            </div>
            <div class="model-card {'active' if is_ls else ''}">
                <div class="model-name">LSTM Neural Net</div>
                <div class="model-pred">₹{lstm_next:,.2f}</div>
                <div class="model-tag {'at' if is_ls else 'it'}">{'✦ Active' if is_ls else (f'Acc {lstm_acc:.1f}%' if lstm_acc else '—')}</div>
            </div>
            <div class="model-card {'active' if is_hy else ''}">
                <div class="model-name">Hybrid Ensemble</div>
                <div class="model-pred">₹{hybrid_next:,.2f}</div>
                <div class="model-tag {'at' if is_hy else 'it'}">{'✦ Active' if is_hy else 'Avg of both'}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        sec("🔬","Backtest — Actual vs Predicted")
        bt_dates = df.index[window_size+split:]
        bfig = go.Figure()
        bfig.add_trace(go.Scatter(x=bt_dates,y=y_true,line=dict(color="#f59e0b",width=2),name="Actual"))
        bfig.add_trace(go.Scatter(x=bt_dates,y=lr_preds,line=dict(color="#6366f1",width=1.5,dash="dash"),name="LR"))
        if lstm_model:
            bfig.add_trace(go.Scatter(x=bt_dates,y=lstm_preds,line=dict(color="#34d399",width=1.5,dash="dot"),name="LSTM"))
        bfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                           height=300,font=dict(family="Space Grotesk",color="#64748b",size=11),
                           legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,bgcolor="rgba(0,0,0,0)"),
                           xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix="₹",tickformat=",.0f"),
                           margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(bfig, use_container_width=True)
        mc1,mc2,mc3,mc4 = st.columns(4)
        mc1.metric("LR MAE",      f"₹{lr_mae:,.0f}")
        mc2.metric("LR RMSE",     f"₹{lr_rmse:,.0f}")
        mc3.metric("LR Accuracy", f"{lr_acc:.1f}%")
        if lstm_acc: mc4.metric("LSTM Accuracy", f"{lstm_acc:.1f}%")

    # SIGNAL BANNER
    sec("🔮","Next Candle Prediction")
    if   predicted_price > current_price*1.02:
        sig,sc,si,sd = "BUY","buy","📈",  f"{symbol} shows upward momentum. Model projects a {change_percent:.2f}% gain."
    elif predicted_price < current_price*0.98:
        sig,sc,si,sd = "SELL","sell","📉", f"{symbol} shows downward pressure. Model projects a {abs(change_percent):.2f}% decline."
    else:
        sig,sc,si,sd = "HOLD","hold","⚖️", f"{symbol} is consolidating. Price within ±2%. Wait for a clearer signal."

    st.markdown(f"""
    <div class="signal-banner {sc}">
        <div class="sig-icon">{si}</div>
        <div><div class="sig-title {sc}">{sig} Signal — {model_choice}</div><div class="sig-desc">{sd}</div></div>
        <div style="margin-left:auto;text-align:right">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.4rem;color:#f1f5f9;font-weight:600">₹{predicted_price:,.2f}</div>
            <div style="font-size:.73rem;color:#475569">Predicted Price</div>
        </div>
    </div>""", unsafe_allow_html=True)

    pc1,pc2,pc3,pc4 = st.columns(4)
    pc1.metric("Current Price",   f"₹{current_price:,.2f}")
    pc2.metric("Predicted Price", f"₹{predicted_price:,.2f}")
    pc3.metric("Expected Change", f"{change_percent:+.2f}%")
    pc4.metric("Active Model",    model_choice.split()[0])

    # PORTFOLIO
    sec("💼","Portfolio Simulation")
    coins_bought = investment/current_price
    future_value = coins_bought*predicted_price
    profit       = future_value-investment
    roi          = (profit/investment)*100
    pp1,pp2,pp3,pp4 = st.columns(4)
    pp1.metric("Investment",   f"₹{investment:,.2f}")
    pp2.metric("Coins Bought", f"{coins_bought:.6f} {symbol}")
    pp3.metric("Future Value", f"₹{future_value:,.2f}")
    pp4.metric("P&L",          f"₹{profit:,.2f}", f"{roi:+.2f}%")

    st.markdown("**Scenario Analysis**")
    sc_df = pd.DataFrame({
        "Scenario":       ["Bear (-10%)","Base (Model)","Bull (+10%)","Bull (+20%)"],
        "Price":          [current_price*.9,predicted_price,current_price*1.1,current_price*1.2],
        "Portfolio Value":[coins_bought*current_price*.9,future_value,coins_bought*current_price*1.1,coins_bought*current_price*1.2],
        "P&L":            [coins_bought*current_price*.9-investment,profit,coins_bought*current_price*1.1-investment,coins_bought*current_price*1.2-investment]
    })
    sc_df["Price"]            = sc_df["Price"].apply(lambda x:f"₹{x:,.2f}")
    sc_df["Portfolio Value"]  = sc_df["Portfolio Value"].apply(lambda x:f"₹{x:,.2f}")
    sc_df["P&L"]              = sc_df["P&L"].apply(lambda x:f"₹{x:+,.2f}")
    st.dataframe(sc_df, use_container_width=True, hide_index=True)

    # RISK
    sec("⚠️","Risk Analysis")
    volatility    = df["price"].pct_change().std()
    ann_vol       = volatility*np.sqrt(365)*100
    daily_ret     = df["price"].pct_change().dropna()
    sharpe        = (daily_ret.mean()/daily_ret.std())*np.sqrt(365) if daily_ret.std()>0 else 0
    max_dd        = ((df["price"]/df["price"].cummax())-1).min()*100
    if   volatility<.02: risk_level,risk_cls = "LOW","risk-low"
    elif volatility<.05: risk_level,risk_cls = "MEDIUM","risk-med"
    else:                risk_level,risk_cls = "HIGH","risk-high"

    st.markdown(f'<div class="risk-row"><span style="color:#64748b;font-size:.85rem">Overall Risk:</span><span class="risk-badge {risk_cls}">● {risk_level} RISK</span></div>', unsafe_allow_html=True)
    rr1,rr2,rr3,rr4 = st.columns(4)
    rr1.metric("Daily Volatility",  f"{volatility*100:.2f}%")
    rr2.metric("Annual Volatility", f"{ann_vol:.1f}%")
    rr3.metric("Sharpe Ratio",      f"{sharpe:.2f}")
    rr4.metric("Max Drawdown",      f"{max_dd:.1f}%")

    rfig = go.Figure()
    rfig.add_trace(go.Histogram(x=daily_ret*100,nbinsx=50,marker_color="#6366f1",opacity=.7,name="Daily Returns"))
    rfig.add_vline(x=0,line=dict(color="#f59e0b",width=1.5,dash="dash"))
    rfig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",height=230,
                       title=dict(text="Daily Return Distribution",font=dict(size=12,color="#64748b")),
                       xaxis=dict(title="Return %",gridcolor="#0f172a"),yaxis=dict(title="Frequency",gridcolor="#0f172a"),
                       showlegend=False,margin=dict(l=10,r=10,t=36,b=10))
    st.plotly_chart(rfig, use_container_width=True)

    # INSIGHTS
    sec("📌","AI Market Insights")
    bb_pos  = "near upper band" if df["price"].iloc[-1]>=df["bb_upper"].iloc[-1]*.98 else "near lower band" if df["price"].iloc[-1]<=df["bb_lower"].iloc[-1]*1.02 else "within Bollinger Bands"
    macd_lbl= "MACD above signal — bullish" if df["macd"].iloc[-1]>df["macd_signal"].iloc[-1] else "MACD below signal — bearish"
    ma_lbl  = "above both MA20 & MA50 — uptrend" if current_price>df["ma20"].iloc[-1] and current_price>df["ma50"].iloc[-1] else "below both MAs — downtrend" if current_price<df["ma20"].iloc[-1] and current_price<df["ma50"].iloc[-1] else "between MA20 & MA50 — mixed"
    for icon,txt in [
        ("📊",f"<b>Trend:</b> {symbol} is {ma_lbl}."),
        ("📉",f"<b>Momentum (RSI {rsi_current:.1f}):</b> {rsi_label} — {'watch for reversal.' if rsi_current>65 or rsi_current<35 else 'no extreme signal.'}"),
        ("🎯",f"<b>Bollinger Bands:</b> Price is {bb_pos}."),
        ("⚡",f"<b>MACD:</b> {macd_lbl}."),
        ("🛡️",f"<b>Risk:</b> {risk_level} ({ann_vol:.1f}% annual vol). Sharpe {sharpe:.2f}.")
    ]:
        st.markdown(f'<div class="insight-pill"><span class="ic">{icon}</span><span>{txt}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="footer">Built by <span>Nagesh</span> · CryptoSage AI v3.0 · Model: {model_choice} · <span>Not financial advice</span></div>', unsafe_allow_html=True)


# ===================================================
# ============  VIEW: MANUAL MODEL  =================
# ===================================================
elif VIEW == "Manual Model":

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-badge">Zero API · Fully Local · No Internet Required</div>
        <p class="hero-title">🧠 <span>Manual Model</span> Lab</p>
        <p class="hero-sub">Build, train & test prediction models on synthetic or uploaded data — no API key needed</p>
    </div>""", unsafe_allow_html=True)

    # ---- TABS inside manual model ----
    tab1, tab2, tab3 = st.tabs(["📐 Build & Train", "📊 Visualise Results", "🔮 Predict"])

    # ---- shared state ----
    if "mm_trained" not in st.session_state:
        st.session_state.mm_trained      = False
        st.session_state.mm_model_type   = "Linear Regression"
        st.session_state.mm_results      = {}
        st.session_state.mm_df           = None

    # =====================  TAB 1: BUILD  =====================
    with tab1:
        st.markdown('<div class="mm-card"><div class="mm-title">Step 1 — Data Source</div><div class="mm-desc">Choose synthetic data (no internet) or upload your own CSV.</div>', unsafe_allow_html=True)

        data_src = st.radio("Data Source", ["🎲 Synthetic (Sine + Noise)", "📂 Upload CSV"], horizontal=True)

        if data_src == "🎲 Synthetic (Sine + Noise)":
            c1,c2,c3 = st.columns(3)
            n_points  = c1.slider("Data Points",   100, 2000, 500)
            trend     = c2.selectbox("Trend",       ["Uptrend","Downtrend","Sideways","Volatile"])
            noise_lvl = c3.slider("Noise Level",    0.01, 0.5, 0.1, 0.01)
            base      = 50000.0
            t         = np.linspace(0, 4*np.pi, n_points)
            if   trend=="Uptrend":   prices_raw = base + base*0.4*(t/(4*np.pi)) + base*0.08*np.sin(t*3)
            elif trend=="Downtrend": prices_raw = base + base*0.4*(1 - t/(4*np.pi)) + base*0.08*np.sin(t*3)
            elif trend=="Sideways":  prices_raw = base + base*0.1*np.sin(t*2)
            else:                    prices_raw = base + base*0.15*np.sin(t*5)*np.cos(t*2)
            prices_raw += prices_raw * noise_lvl * np.random.randn(n_points)
            prices_raw  = np.abs(prices_raw)
            dates        = pd.date_range(end=pd.Timestamp.today(), periods=n_points, freq="h")
            mm_df        = pd.DataFrame({"price": prices_raw}, index=dates)
            st.session_state.mm_df = mm_df
            st.success(f"✔ Generated {n_points} synthetic data points · {trend}")
        else:
            uploaded = st.file_uploader("Upload CSV (must have a 'price' column)", type=["csv"])
            if uploaded:
                mm_df = pd.read_csv(uploaded)
                if "price" not in mm_df.columns:
                    st.error("CSV must contain a column named 'price'"); mm_df = None
                else:
                    mm_df.index = pd.to_datetime(mm_df.index) if mm_df.index.dtype!="object" else pd.RangeIndex(len(mm_df))
                    st.session_state.mm_df = mm_df
                    st.success(f"✔ Loaded {len(mm_df)} rows")
            mm_df = st.session_state.mm_df

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Model config ----
        st.markdown('<div class="mm-card"><div class="mm-title">Step 2 — Model Configuration</div><div class="mm-desc">All parameters are manual — no external libraries needed beyond scikit-learn and keras.</div>', unsafe_allow_html=True)

        m1,m2 = st.columns(2)
        mm_model_type = m1.selectbox(
            "Model Type",
            ["Linear Regression","LSTM Neural Net","Simple Moving Average (SMA)","Exponential Smoothing (EMA)","Polynomial Regression"]
        )
        window_mm = m2.slider("Lookback Window", 5, 60, 15)

        if mm_model_type == "LSTM Neural Net":
            l1,l2,l3,l4 = st.columns(4)
            lstm_units1 = l1.slider("LSTM Units Layer 1", 16, 128, 64, 16)
            lstm_units2 = l2.slider("LSTM Units Layer 2", 8,  64,  32, 8)
            dropout_r   = l3.slider("Dropout Rate",       0.0, 0.5, 0.2, 0.05)
            epochs_mm   = l4.slider("Epochs",             3, 30, 10)
        elif mm_model_type == "Polynomial Regression":
            poly_deg = st.slider("Polynomial Degree", 2, 5, 2)

        train_split = st.slider("Train / Test Split %", 60, 90, 80)
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- TRAIN button ----
        if st.button("🚀 Train Model", use_container_width=True, type="primary"):
            mm_df = st.session_state.mm_df
            if mm_df is None:
                st.error("Please select or upload data first.")
            else:
                with st.spinner("Training model on your data…"):
                    prices_mm  = mm_df["price"].values.reshape(-1,1)
                    sc_mm      = MinMaxScaler()
                    scaled_mm  = sc_mm.fit_transform(prices_mm)

                    X_mm,y_mm = [],[]
                    for i in range(window_mm,len(scaled_mm)):
                        X_mm.append(scaled_mm[i-window_mm:i]); y_mm.append(scaled_mm[i])
                    X_mm = np.array(X_mm); y_mm = np.array(y_mm)

                    sp       = int(len(X_mm)*train_split/100)
                    Xtr,Xte  = X_mm[:sp],X_mm[sp:]
                    ytr,yte  = y_mm[:sp],y_mm[sp:]

                    # ---- choose model ----
                    trained_model = None
                    if mm_model_type == "Linear Regression":
                        trained_model = LinearRegression()
                        trained_model.fit(Xtr.reshape(Xtr.shape[0],-1), ytr)
                        preds_sc = trained_model.predict(Xte.reshape(Xte.shape[0],-1)).flatten()

                    elif mm_model_type == "Polynomial Regression":
                        from sklearn.preprocessing import PolynomialFeatures
                        from sklearn.pipeline import Pipeline
                        trained_model = Pipeline([
                            ("poly", PolynomialFeatures(degree=poly_deg)),
                            ("lr",   LinearRegression())
                        ])
                        trained_model.fit(Xtr.reshape(Xtr.shape[0],-1), ytr.flatten())
                        preds_sc = trained_model.predict(Xte.reshape(Xte.shape[0],-1)).flatten()

                    elif mm_model_type == "LSTM Neural Net":
                        trained_model = Sequential([
                            LSTM(lstm_units1,return_sequences=True,input_shape=(window_mm,1)),
                            Dropout(dropout_r),
                            LSTM(lstm_units2),
                            Dropout(dropout_r),
                            Dense(16), Dense(1)
                        ])
                        trained_model.compile(optimizer="adam",loss="huber")
                        trained_model.fit(Xtr,ytr,epochs=epochs_mm,batch_size=32,verbose=0)
                        preds_sc = trained_model.predict(Xte,verbose=0).flatten()

                    elif mm_model_type == "Simple Moving Average (SMA)":
                        # Pure numpy SMA — zero external ML
                        sma_raw  = pd.Series(prices_mm.flatten()).rolling(window_mm).mean().values
                        sma_sc   = sc_mm.transform(sma_raw.reshape(-1,1)).flatten()
                        preds_sc = sma_sc[window_mm+sp-1:window_mm+sp-1+len(yte)]
                        preds_sc = preds_sc[:len(yte)]
                        trained_model = "SMA"

                    elif mm_model_type == "Exponential Smoothing (EMA)":
                        alpha     = 0.3
                        ema_vals  = [prices_mm[0][0]]
                        for v in prices_mm[1:]:
                            ema_vals.append(alpha*v[0]+(1-alpha)*ema_vals[-1])
                        ema_sc   = sc_mm.transform(np.array(ema_vals).reshape(-1,1)).flatten()
                        preds_sc = ema_sc[window_mm+sp:window_mm+sp+len(yte)]
                        preds_sc = preds_sc[:len(yte)]
                        trained_model = "EMA"

                    # pad if needed
                    min_len  = min(len(preds_sc),len(yte))
                    preds_sc = preds_sc[:min_len]
                    yte_use  = yte[:min_len]

                    preds_true = sc_mm.inverse_transform(preds_sc.reshape(-1,1)).flatten()
                    y_true_mm  = sc_mm.inverse_transform(yte_use.reshape(-1,1)).flatten()

                    mae_mm  = mean_absolute_error(y_true_mm,preds_true)
                    rmse_mm = np.sqrt(mean_squared_error(y_true_mm,preds_true))
                    mape_mm = np.mean(np.abs((y_true_mm-preds_true)/np.maximum(y_true_mm,1e-8)))*100
                    acc_mm  = max(0,100-mape_mm)

                    # Next-step prediction
                    last_w_mm = scaled_mm[-window_mm:]
                    if mm_model_type in ["Linear Regression","Polynomial Regression"]:
                        next_sc = trained_model.predict(last_w_mm.reshape(1,-1))[0]
                    elif mm_model_type == "LSTM Neural Net":
                        next_sc = trained_model.predict(last_w_mm.reshape(1,window_mm,1),verbose=0)[0][0]
                    elif mm_model_type == "Simple Moving Average (SMA)":
                        next_sc = float(np.mean(scaled_mm[-window_mm:]))
                    else:  # EMA
                        alpha   = 0.3
                        ev      = scaled_mm[-window_mm][0]
                        for v in scaled_mm[-window_mm+1:]:
                            ev = alpha*v[0]+(1-alpha)*ev
                        next_sc = float(ev)

                    next_price = sc_mm.inverse_transform([[next_sc]])[0][0]

                    st.session_state.mm_trained    = True
                    st.session_state.mm_model_type = mm_model_type
                    st.session_state.mm_results    = {
                        "mae":mae_mm,"rmse":rmse_mm,"mape":mape_mm,"acc":acc_mm,
                        "preds":preds_true,"y_true":y_true_mm,
                        "next_price":next_price,"current_price":mm_df["price"].iloc[-1],
                        "dates":mm_df.index[window_mm+sp:window_mm+sp+min_len],
                        "full_df":mm_df
                    }
                    st.success(f"✅ {mm_model_type} trained! Accuracy: {acc_mm:.1f}%")

    # =====================  TAB 2: VISUALISE  =====================
    with tab2:
        if not st.session_state.mm_trained:
            st.info("Train a model in the **Build & Train** tab first.")
        else:
            res     = st.session_state.mm_results
            mm_type = st.session_state.mm_model_type

            # Accuracy KPIs
            sec("📋","Model Performance")
            kc = (
                kpi_html("Accuracy",  f"{res['acc']:.1f}%",  None, "linear-gradient(90deg,#34d399,#6ee7b7)") +
                kpi_html("MAE",       f"₹{res['mae']:,.0f}", None, "linear-gradient(90deg,#6366f1,#818cf8)") +
                kpi_html("RMSE",      f"₹{res['rmse']:,.0f}",None, "linear-gradient(90deg,#f59e0b,#fbbf24)") +
                kpi_html("MAPE",      f"{res['mape']:.2f}%", None, "linear-gradient(90deg,#a78bfa,#c4b5fd)")
            )
            st.markdown(f'<div class="kpi-grid">{kc}</div>', unsafe_allow_html=True)

            # Full history chart
            sec("📈","Full Price History (Synthetic / Uploaded)")
            full_fig = go.Figure()
            full_fig.add_trace(go.Scatter(
                x=res["full_df"].index, y=res["full_df"]["price"],
                mode="lines", line=dict(color="#f59e0b",width=1.8),
                fill="tozeroy", fillcolor="rgba(245,158,11,.05)", name="Price"
            ))
            full_fig.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                height=280,font=dict(family="Space Grotesk",color="#64748b",size=11),
                xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix="₹",tickformat=",.0f"),
                showlegend=False,margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(full_fig, use_container_width=True)

            # Actual vs Predicted
            sec("🔬",f"Actual vs Predicted — {mm_type}")
            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Scatter(x=res["dates"],y=res["y_true"],line=dict(color="#f59e0b",width=2),name="Actual"))
            cmp_fig.add_trace(go.Scatter(x=res["dates"],y=res["preds"], line=dict(color="#6366f1",width=1.8,dash="dash"),name="Predicted"))
            cmp_fig.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                height=320,font=dict(family="Space Grotesk",color="#64748b",size=11),
                legend=dict(orientation="h",yanchor="bottom",y=1.01,x=0,bgcolor="rgba(0,0,0,0)"),
                xaxis=dict(gridcolor="#0f172a"),yaxis=dict(gridcolor="#0f172a",tickprefix="₹",tickformat=",.0f"),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(cmp_fig, use_container_width=True)

            # Error distribution
            sec("📉","Prediction Error Distribution")
            errors = res["y_true"] - res["preds"]
            err_fig = go.Figure()
            err_fig.add_trace(go.Histogram(x=errors,nbinsx=40,marker_color="#6366f1",opacity=.75,name="Error"))
            err_fig.add_vline(x=0,line=dict(color="#f59e0b",width=1.5,dash="dash"))
            err_fig.update_layout(
                template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(10,10,20,.6)",
                height=220,showlegend=False,
                xaxis=dict(title="Error (₹)",gridcolor="#0f172a"),
                yaxis=dict(title="Frequency",gridcolor="#0f172a"),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(err_fig, use_container_width=True)

    # =====================  TAB 3: PREDICT  =====================
    with tab3:
        if not st.session_state.mm_trained:
            st.info("Train a model in the **Build & Train** tab first.")
        else:
            res     = st.session_state.mm_results
            mm_type = st.session_state.mm_model_type
            cp      = res["current_price"]
            np_     = res["next_price"]
            chg     = ((np_-cp)/cp)*100

            sec("🔮","Next Step Prediction")
            if   np_ > cp*1.02: sc2,si2,sg2 = "buy","📈","BUY"
            elif np_ < cp*0.98: sc2,si2,sg2 = "sell","📉","SELL"
            else:               sc2,si2,sg2 = "hold","⚖️","HOLD"

            st.markdown(f"""
            <div class="signal-banner {sc2}">
                <div class="sig-icon">{si2}</div>
                <div>
                    <div class="sig-title {sc2}">{sg2} Signal · {mm_type}</div>
                    <div class="sig-desc">Trained on local data — no API key used. Next predicted value shown on right.</div>
                </div>
                <div style="margin-left:auto;text-align:right">
                    <div style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;color:#f1f5f9;font-weight:700">₹{np_:,.2f}</div>
                    <div style="font-size:.73rem;color:#475569">Predicted Next Value</div>
                </div>
            </div>""", unsafe_allow_html=True)

            col_a,col_b,col_c = st.columns(3)
            col_a.metric("Current Value",   f"₹{cp:,.2f}")
            col_b.metric("Predicted Value", f"₹{np_:,.2f}")
            col_c.metric("Expected Change", f"{chg:+.2f}%")

            # Accuracy summary
            sec("📋","Model Summary")
            st.markdown(f"""
            <div class="mm-card">
                <div class="mm-title">{mm_type}</div>
                <div class="mm-desc">Trained entirely locally · zero external APIs</div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:8px">
                    <div><div class="kpi-label">Accuracy</div><div class="kpi-value" style="color:#34d399">{res['acc']:.1f}%</div></div>
                    <div><div class="kpi-label">MAE</div><div class="kpi-value">₹{res['mae']:,.0f}</div></div>
                    <div><div class="kpi-label">RMSE</div><div class="kpi-value">₹{res['rmse']:,.0f}</div></div>
                    <div><div class="kpi-label">MAPE</div><div class="kpi-value">{res['mape']:.2f}%</div></div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="footer">Built by <span>Nagesh</span> · CryptoSage AI v3.0 · Manual Model runs 100% locally · <span>Not financial advice</span></div>', unsafe_allow_html=True)
