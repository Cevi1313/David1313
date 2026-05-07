import os
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import schedule
import json
from datetime import datetime, timedelta
from telegram import Bot
from telegram.error import TelegramError

# ---------- CONFIG ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "EURJPY=X", "EURGBP=X", "NZDUSD=X", "USDCHF=X", "EURCHF=X",
    "EURCAD=X", "GBPJPY=X", "GC=F",
    "AUDJPY=X", "GBPCHF=X", "AUDCAD=X", "NZDJPY=X"
]

# ---------- STRATEGY FINAL (hasil backtest XAUUSD H4) ----------
LOOKBACK_SHORT = 80
LOOKBACK_LONG = 160
ADX_THRESHOLD = 12
SL_ATR = 1.5
TP_ATR = 2.5
TIME_STOP_BAR = 20           # 20 bar H4 = 80 jam
MIN_RANGE_ATR = 1.5
USE_EMA_FILTER = False       # EMA50 tidak dipakai

# ---------- SCAN ----------
SCAN_INTERVAL_HOURS = 2
TELEGRAM_DELAY_SEC = 1
STATE_FILE = "trading_state.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
bot = Bot(token=TELEGRAM_TOKEN)

# ---------- HELPER ----------
def get_pip_size(symbol):
    sym = symbol.upper()
    if any(x in sym for x in ('GC', 'XAU', 'GOLD', 'SI', 'XAG')):
        return 0.01
    elif 'JPY' in sym:
        return 0.01
    else:
        return 0.0001

def price_to_pip(symbol, distance):
    return distance / get_pip_size(symbol)

def fetch_h4(symbol, days=30):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1h", progress=False, timeout=30)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [c.lower() for c in df.columns]
        df = df[['open','high','low','close','volume']]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df4 = df.resample('4h').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna()
        df4.columns = ['Open','High','Low','Close','Vol']
        return df4 if len(df4) >= LOOKBACK_LONG else None
    except Exception as e:
        logging.error(f"Fetch error {symbol}: {e}")
        return None

def calc_sr(window):
    h = window['High'].values
    l = window['Low'].values
    sh, sl = [], []
    for j in range(2, len(window)-2):
        if h[j] > h[j-1] and h[j] > h[j-2] and h[j] > h[j+1] and h[j] > h[j+2]:
            sh.append(h[j])
        if l[j] < l[j-1] and l[j] < l[j-2] and l[j] < l[j+1] and l[j] < l[j+2]:
            sl.append(l[j])
    if not sh or not sl:
        return np.nan, np.nan
    return np.mean(sorted(sl)[:3]), np.mean(sorted(sh)[-3:])

def get_support_resistance(df4):
    sup_short = pd.Series(np.nan, index=df4.index)
    res_short = pd.Series(np.nan, index=df4.index)
    sup_long = pd.Series(np.nan, index=df4.index)
    res_long = pd.Series(np.nan, index=df4.index)
    for i in range(LOOKBACK_LONG, len(df4)):
        w_short = df4.iloc[i-LOOKBACK_SHORT:i]
        w_long = df4.iloc[i-LOOKBACK_LONG:i]
        sup_short.iloc[i], res_short.iloc[i] = calc_sr(w_short)
        sup_long.iloc[i], res_long.iloc[i] = calc_sr(w_long)
    sup_short.ffill(inplace=True)
    res_short.ffill(inplace=True)
    sup_long.ffill(inplace=True)
    res_long.ffill(inplace=True)
    support = np.maximum(sup_short, sup_long)
    resistance = np.minimum(res_short, res_long)
    return support, resistance

def calculate_indicators(df4):
    close = df4['Close']
    high = df4['High']
    low = df4['Low']
    tr = np.maximum(high-low,
                    np.maximum(abs(high-close.shift()), abs(low-close.shift())))
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().multiply(-1).clip(lower=0)
    atr_dx = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_dx)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_dx)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()  # tetap dihitung tapi tidak digunakan kalau USE_EMA_FILTER False
    return {'close':close,'high':high,'low':low,'open':df4['Open'],
            'volume':df4['Vol'],'ema50':ema50,'atr14':atr14,'adx':adx}

def detect_signals(df4, ind, support, resistance, symbol):
    close = ind['close']
    high = ind['high']
    low = ind['low']
    open_ = ind['open']
    ema50 = ind['ema50']
    atr14 = ind['atr14']
    adx = ind['adx']

    strong = adx > ADX_THRESHOLD
    min_range = atr14 * MIN_RANGE_ATR
    atr_buf = atr14 * 1.5
    bullish = close > open_
    bearish = close < open_

    if USE_EMA_FILTER:
        trend_up = close > ema50
        trend_down = close < ema50
    else:
        trend_up = True
        trend_down = True

    # Breakout
    buy_break = (close > resistance) & trend_up & strong & ((resistance - support) >= min_range)
    sell_break = (close < support) & trend_down & strong & ((resistance - support) >= min_range)

    # Bounce
    buy_bounce = (low <= support + atr_buf) & (close > support) & bullish & trend_up & strong & ((resistance - support) >= min_range)
    sell_bounce = (high >= resistance - atr_buf) & (close < resistance) & bearish & trend_down & strong & ((resistance - support) >= min_range)

    # Gabungan
    buy_sig = buy_break | buy_bounce
    sell_sig = sell_break | sell_bounce

    if buy_sig.iloc[-1] or sell_sig.iloc[-1]:
        entry = close.iloc[-1]
        atr_val = atr14.iloc[-1]
        if pd.isna(atr_val) or atr_val == 0:
            return []
        sl_dist = SL_ATR * atr_val
        tp_dist = TP_ATR * atr_val
        signals = []
        if buy_sig.iloc[-1]:
            signals.append({
                'type':'BUY','entry':round(entry,5),
                'sl':round(entry-sl_dist,5),'tp':round(entry+tp_dist,5),
                'sl_pip':round(price_to_pip(symbol, sl_dist),1),
                'tp_pip':round(price_to_pip(symbol, tp_dist),1),
                'rr':round(TP_ATR/SL_ATR,2),
                'support':round(support.iloc[-1],5),
                'resistance':round(resistance.iloc[-1],5)
            })
        if sell_sig.iloc[-1]:
            signals.append({
                'type':'SELL','entry':round(entry,5),
                'sl':round(entry+sl_dist,5),'tp':round(entry-tp_dist,5),
                'sl_pip':round(price_to_pip(symbol, sl_dist),1),
                'tp_pip':round(price_to_pip(symbol, tp_dist),1),
                'rr':round(TP_ATR/SL_ATR,2),
                'support':round(support.iloc[-1],5),
                'resistance':round(resistance.iloc[-1],5)
            })
        return signals
    return []

# ---------- STATE ANTI DUPLIKASI ----------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def is_duplicate(symbol, sig):
    state = load_state()
    key = f"{symbol}_{sig['type']}_{sig['support']}_{sig['resistance']}"
    last = state.get(key)
    if last:
        last_time = datetime.fromisoformat(last)
        if datetime.now() - last_time < timedelta(hours=24):
            return True
    return False

def mark_sent(symbol, sig):
    state = load_state()
    key = f"{symbol}_{sig['type']}_{sig['support']}_{sig['resistance']}"
    state[key] = datetime.now().isoformat()
    save_state(state)

# ---------- TELEGRAM ----------
def send_telegram_message(text):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text, parse_mode='Markdown')
        logging.info("Message sent to Telegram")
        time.sleep(TELEGRAM_DELAY_SEC)
    except TelegramError as e:
        logging.error(f"Telegram error: {e}")

def send_alert(symbol, signal):
    msg = (
        f"🔔 **SINYAL TRADING** 🔔\n"
        f"Symbol: {symbol}\n"
        f"Signal: {signal['type']}\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['sl']} ({signal['sl_pip']} pip)\n"
        f"TP: {signal['tp']} ({signal['tp_pip']} pip)\n"
        f"RR: 1:{signal['rr']}\n"
        f"Support: {signal['support']} | Resistance: {signal['resistance']}\n"
        f"Time Stop: {TIME_STOP_BAR} bar (H4)\n"
        f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_telegram_message(msg)

# ---------- SCAN ----------
def scan_symbol(symbol):
    logging.info(f"Scanning {symbol}...")
    df4 = fetch_h4(symbol, days=30)
    if df4 is None or len(df4) < LOOKBACK_LONG:
        logging.warning(f"Data tidak cukup untuk {symbol}")
        return
    support, resistance = get_support_resistance(df4)
    if support.iloc[-1] is np.nan or resistance.iloc[-1] is np.nan:
        logging.warning(f"S/R invalid {symbol}")
        return
    ind = calculate_indicators(df4)
    if pd.isna(ind['adx'].iloc[-1]):
        logging.warning(f"Indikator tidak lengkap {symbol}")
        return
    signals = detect_signals(df4, ind, support, resistance, symbol)
    if signals:
        for sig in signals:
            if is_duplicate(symbol, sig):
                logging.info(f"{symbol}: Duplicate ignored")
                continue
            mark_sent(symbol, sig)
            send_alert(symbol, sig)
            print(f"\n🔔 {symbol} - {sig['type']} @ {sig['entry']} | SL={sig['sl']} TP={sig['tp']} | RR=1:{sig['rr']}")
    else:
        logging.info(f"{symbol}: No signal")

def scan_all():
    logging.info("=== SCAN MULTI-PAIR DIMULAI ===")
    for sym in SYMBOLS:
        try:
            scan_symbol(sym)
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error {sym}: {e}")
    logging.info("=== SCAN SELESAI ===")

def send_startup():
    msg = (
        f"🤖 **AGEN TRADING AKTIF**\n"
        f"Scan setiap {SCAN_INTERVAL_HOURS} jam\n"
        f"Aturan: ADX>{ADX_THRESHOLD}, Lookback S/R {LOOKBACK_SHORT}/{LOOKBACK_LONG}, "
        f"Breakout+Bounce, SL={SL_ATR}ATR, TP={TP_ATR}ATR, TimeStop={TIME_STOP_BAR}bar\n"
        f"EMA filter: {'AKTIF' if USE_EMA_FILTER else 'NONAKTIF'}"
    )
    send_telegram_message(msg)

# ---------- MAIN ----------
if __name__ == "__main__":
    logging.info("🤖 Bot trading final dijalankan.")
    send_startup()
    scan_all()
    schedule.every(SCAN_INTERVAL_HOURS).hours.do(scan_all)
    while True:
        schedule.run_pending()
        time.sleep(60)
