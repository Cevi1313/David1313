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

# ================= KONFIGURASI =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 7 pair terbaik hasil backtest 2015-2026
SYMBOLS = [
    "GC=F",        # Gold
    "USDJPY=X",
    "NZDJPY=X",
    "CHFJPY=X",
    "USDCAD=X",
    "CADJPY=X",
    "AUDJPY=X"
]

# Parameter optimal per pair (Lookback, ADX, Cap, Time Stop)
PARAMS = {
    "GC=F":     {'lb': (120, 240), 'adx': 12, 'cap': None, 'ts': 20},
    "USDJPY=X": {'lb': (60, 120),  'adx': 16, 'cap': 500,  'ts': None},
    "NZDJPY=X": {'lb': (80, 160),  'adx': 12, 'cap': None, 'ts': 20},
    "CHFJPY=X": {'lb': (80, 160),  'adx': 12, 'cap': None, 'ts': 20},
    "USDCAD=X": {'lb': (80, 160),  'adx': 12, 'cap': None, 'ts': 20},
    "CADJPY=X": {'lb': (80, 160),  'adx': 12, 'cap': None, 'ts': 20},
    "AUDJPY=X": {'lb': (80, 160),  'adx': 12, 'cap': None, 'ts': 20},
}

SCAN_INTERVAL_HOURS = 2
TELEGRAM_DELAY_SEC = 1
STATE_FILE = "trading_state.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
bot = Bot(token=TELEGRAM_TOKEN)

# ================= FUNGSI-FUNGSI =================
def get_pip_size(symbol):
    if 'JPY' in symbol:
        return 0.01
    elif 'GC' in symbol:
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
        return df4
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

def get_support_resistance(df4, lb_short, lb_long):
    support = pd.Series(np.nan, index=df4.index)
    resistance = pd.Series(np.nan, index=df4.index)
    for i in range(lb_long, len(df4)):
        w_short = df4.iloc[i-lb_short:i]
        w_long = df4.iloc[i-lb_long:i]
        sup_s, res_s = calc_sr(w_short)
        sup_l, res_l = calc_sr(w_long)
        support.iloc[i] = np.nanmax([sup_s, sup_l])
        resistance.iloc[i] = np.nanmin([res_s, res_l])
    support.ffill(inplace=True)
    resistance.ffill(inplace=True)
    return support, resistance

def calculate_indicators(df4):
    close = df4['Close']
    high = df4['High']
    low = df4['Low']
    tr = np.maximum(high-low, np.maximum(abs(high-close.shift()), abs(low-close.shift())))
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().multiply(-1).clip(lower=0)
    atr_dx = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_dx)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_dx)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    return {'close':close, 'high':high, 'low':low, 'open':df4['Open'],
            'atr14':atr14, 'adx':adx}

def detect_signals(df4, ind, support, resistance, symbol):
    cfg = PARAMS.get(symbol, PARAMS["AUDJPY=X"])
    adx_th = cfg['adx']
    cap_pips = cfg['cap']
    time_stop = cfg['ts']
    
    close = ind['close']
    high = ind['high']
    low = ind['low']
    open_ = ind['open']
    atr14 = ind['atr14']
    adx = ind['adx']

    strong = adx > adx_th
    min_range = atr14 * 1.5
    atr_buf = atr14 * 1.5
    bullish = close > open_
    bearish = close < open_

    buy_break = (close > resistance) & strong & ((resistance - support) >= min_range)
    sell_break = (close < support) & strong & ((resistance - support) >= min_range)
    buy_bounce = (low <= support + atr_buf) & (close > support) & bullish & strong & ((resistance - support) >= min_range)
    sell_bounce = (high >= resistance - atr_buf) & (close < resistance) & bearish & strong & ((resistance - support) >= min_range)

    buy_sig = buy_break | buy_bounce
    sell_sig = sell_break | sell_bounce

    if not (buy_sig.iloc[-1] or sell_sig.iloc[-1]):
        return []

    entry = close.iloc[-1]
    atr_val = atr14.iloc[-1]
    if pd.isna(atr_val) or atr_val == 0:
        return []

    sl_dist = 1.5 * atr_val
    tp_dist = 2.5 * atr_val
    pip_size = get_pip_size(symbol)
    if cap_pips:
        max_dist = cap_pips * pip_size
        sl_dist = min(sl_dist, max_dist)
        tp_dist = min(tp_dist, max_dist)

    signals = []
    if buy_sig.iloc[-1]:
        signals.append({
            'type': 'BUY',
            'entry': round(entry, 5),
            'sl': round(entry - sl_dist, 5),
            'tp': round(entry + tp_dist, 5),
            'sl_pip': round(price_to_pip(symbol, sl_dist), 1),
            'tp_pip': round(price_to_pip(symbol, tp_dist), 1),
            'rr': round(tp_dist/sl_dist, 2) if sl_dist > 0 else 0,
            'support': round(support.iloc[-1], 5),
            'resistance': round(resistance.iloc[-1], 5),
            'time_stop': time_stop
        })
    if sell_sig.iloc[-1]:
        signals.append({
            'type': 'SELL',
            'entry': round(entry, 5),
            'sl': round(entry + sl_dist, 5),
            'tp': round(entry - tp_dist, 5),
            'sl_pip': round(price_to_pip(symbol, sl_dist), 1),
            'tp_pip': round(price_to_pip(symbol, tp_dist), 1),
            'rr': round(tp_dist/sl_dist, 2) if sl_dist > 0 else 0,
            'support': round(support.iloc[-1], 5),
            'resistance': round(resistance.iloc[-1], 5),
            'time_stop': time_stop
        })
    return signals

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
def kirim_telegram(pesan):
    try:
        # Hapus parse_mode agar tidak error pada format tertentu
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=pesan)
        logging.info("✅ Notifikasi terkirim.")
        time.sleep(TELEGRAM_DELAY_SEC)
    except TelegramError as e:
        logging.error(f"Telegram error: {e}")
    except Exception as e:
        logging.error(f"Error umum: {e}")

def send_startup():
    pesan = "🤖 **Bot Sinyal Aktif**\nScan 7 pair setiap 2 jam.\nParameter optimal per pair (backtest 2015‑2026)."
    kirim_telegram(pesan)

def send_alert(symbol, signal):
    ts = signal['time_stop']
    ts_str = f"{ts} bar H4" if ts else "Tidak ada"
    msg = (
        f"🔔 **SINYAL TRADING**\n"
        f"Symbol: {symbol}\n"
        f"Signal: {signal['type']}\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['sl']} ({signal['sl_pip']} pip)\n"
        f"TP: {signal['tp']} ({signal['tp_pip']} pip)\n"
        f"RR: 1:{signal['rr']}\n"
        f"Support: {signal['support']} | Resistance: {signal['resistance']}\n"
        f"Time Stop: {ts_str}\n"
        f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    kirim_telegram(msg)

# ---------- SCAN ----------
def scan_symbol(symbol):
    logging.info(f"Scanning {symbol}...")
    df4 = fetch_h4(symbol, days=30)
    if df4 is None or len(df4) < 160:
        logging.warning(f"Data tidak cukup {symbol}")
        return
    cfg = PARAMS.get(symbol, PARAMS["AUDJPY=X"])
    lb_short, lb_long = cfg['lb']
    support, resistance = get_support_resistance(df4, lb_short, lb_long)
    if pd.isna(support.iloc[-1]) or pd.isna(resistance.iloc[-1]):
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
    else:
        logging.info(f"{symbol}: No signal")

def scan_all():
    logging.info("=== SCAN DIMULAI ===")
    for sym in SYMBOLS:
        try:
            scan_symbol(sym)
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error {sym}: {e}")
    logging.info("=== SCAN SELESAI ===")

# ---------- MAIN ----------
def main():
      # <-- hapus setelah yakin token benar
    send_startup()
    scan_all()
    schedule.every(SCAN_INTERVAL_HOURS).hours.do(scan_all)
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
