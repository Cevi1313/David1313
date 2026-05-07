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

# ---------- 7 PAIR TERBAIK ----------
SYMBOLS = [
    "GC=F",        # Gold
    "USDJPY=X",    # USD/JPY
    "NZDJPY=X",    # NZD/JPY
    "CHFJPY=X",    # CHF/JPY
    "USDCAD=X",    # USD/CAD
    "CADJPY=X",    # CAD/JPY
    "AUDJPY=X"     # AUD/JPY
]

# ---------- PARAMETER OPTIMAL PER PAIR ----------
PAIR_PARAMS = {
    "GC=F":       {'lb_short': 120, 'lb_long': 240, 'adx': 12, 'cap': None, 'time_stop': 20},
    "USDJPY=X":   {'lb_short': 60,  'lb_long': 120, 'adx': 16, 'cap': 500, 'time_stop': None},
    "NZDJPY=X":   {'lb_short': 80,  'lb_long': 160, 'adx': 12, 'cap': None, 'time_stop': 20},
    "CHFJPY=X":   {'lb_short': 80,  'lb_long': 160, 'adx': 12, 'cap': None, 'time_stop': 20},
    "USDCAD=X":   {'lb_short': 80,  'lb_long': 160, 'adx': 12, 'cap': None, 'time_stop': 20},
    "CADJPY=X":   {'lb_short': 80,  'lb_long': 160, 'adx': 12, 'cap': None, 'time_stop': 20},
    "AUDJPY=X":   {'lb_short': 80,  'lb_long': 160, 'adx': 12, 'cap': None, 'time_stop': 20}
}

# Default parameter jika pair tidak ada di PAIR_PARAMS
DEFAULT_LB_SHORT = 80
DEFAULT_LB_LONG = 160
DEFAULT_ADX = 12
DEFAULT_CAP = None
DEFAULT_TIME_STOP = 20

# ---------- SCAN ----------
SCAN_INTERVAL_HOURS = 2
TELEGRAM_DELAY_SEC = 1
STATE_FILE = "trading_state.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
bot = Bot(token=TELEGRAM_TOKEN)

# ---------- FUNGSI PIP ----------
def get_pip_size(symbol):
    if 'JPY' in symbol:
        return 0.01
    elif symbol == 'GC=F':
        return 0.01
    else:
        return 0.0001

def price_to_pip(symbol, distance):
    return distance / get_pip_size(symbol)

# ---------- AMBIL DATA H4 ----------
def fetch_h4_data(symbol, days_back=30):
    try:
        df = yf.download(symbol, period=f"{days_back}d", interval="1h", progress=False, timeout=30)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [c.lower() for c in df.columns]
        df = df[['open','high','low','close','volume']]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df_4h = df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        df_4h.columns = ['Open','High','Low','Close','Vol']
        return df_4h
    except Exception as e:
        logging.error(f"Error fetching {symbol}: {e}")
        return None

# ---------- SUPPORT / RESISTANCE ----------
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

def get_sr(df4, lb_s, lb_l):
    support = pd.Series(np.nan, index=df4.index)
    resistance = pd.Series(np.nan, index=df4.index)
    for i in range(lb_l, len(df4)):
        w_short = df4.iloc[i-lb_s:i]
        w_long = df4.iloc[i-lb_l:i]
        sup_s, res_s = calc_sr(w_short)
        sup_l, res_l = calc_sr(w_long)
        support.iloc[i] = np.nanmax([sup_s, sup_l])
        resistance.iloc[i] = np.nanmin([res_s, res_l])
    support.ffill(inplace=True)
    resistance.ffill(inplace=True)
    return support, resistance

# ---------- INDIKATOR ----------
def calculate_indicators(df4):
    close = df4['Close']
    high = df4['High']
    low = df4['Low']
    tr = np.maximum(high - low,
                    np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().multiply(-1).clip(lower=0)
    atr_dx = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_dx)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr_dx)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    adx = dx.ewm(alpha=1/14, adjust=False).mean()
    return {
        'close': close, 'high': high, 'low': low, 'open': df4['Open'],
        'atr14': atr14, 'adx': adx
    }

# ---------- DETEKSI SINYAL ----------
def detect_signals(df4, ind, support, resistance, symbol, params):
    close = ind['close']
    high = ind['high']
    low = ind['low']
    open_ = ind['open']
    atr14 = ind['atr14']
    adx = ind['adx']

    adx_th = params['adx']
    cap = params['cap']
    time_stop = params['time_stop']

    strong = adx > adx_th
    min_range = atr14 * 1.5
    atr_buf = atr14 * 1.5
    bullish = close > open_
    bearish = close < open_

    # Breakout + Bounce
    buy_break = (close > resistance) & strong & ((resistance - support) >= min_range)
    sell_break = (close < support) & strong & ((resistance - support) >= min_range)
    buy_bounce = (low <= support + atr_buf) & (close > support) & bullish & strong & ((resistance - support) >= min_range)
    sell_bounce = (high >= resistance - atr_buf) & (close < resistance) & bearish & strong & ((resistance - support) >= min_range)

    buy_signal = buy_break | buy_bounce
    sell_signal = sell_break | sell_bounce

    # Hanya sinyal terbaru
    if buy_signal.iloc[-1] or sell_signal.iloc[-1]:
        entry = close.iloc[-1]
        atr_val = atr14.iloc[-1]
        if pd.isna(atr_val) or atr_val == 0:
            return []

        pip_size = get_pip_size(symbol)
        sl_dist = 1.5 * atr_val
        tp_dist = 2.5 * atr_val
        if cap:
            max_d = cap * pip_size
            sl_dist = min(sl_dist, max_d)
            tp_dist = min(tp_dist, max_d)

        signals = []
        if buy_signal.iloc[-1]:
            signals.append({
                'type': 'BUY',
                'entry': round(entry, 5),
                'sl': round(entry - sl_dist, 5),
                'tp': round(entry + tp_dist, 5),
                'rr': round(2.5/1.5, 2),
                'support': round(support.iloc[-1], 5),
                'resistance': round(resistance.iloc[-1], 5),
                'time_stop': time_stop
            })
        if sell_signal.iloc[-1]:
            signals.append({
                'type': 'SELL',
                'entry': round(entry, 5),
                'sl': round(entry + sl_dist, 5),
                'tp': round(entry - tp_dist, 5),
                'rr': round(2.5/1.5, 2),
                'support': round(support.iloc[-1], 5),
                'resistance': round(resistance.iloc[-1], 5),
                'time_stop': time_stop
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
    ts = signal['time_stop']
    ts_str = f"{ts} bar (H4)" if ts else "Tidak ada"
    msg = (
        f"🔔 **SINYAL TRADING** 🔔\n"
        f"Symbol: {symbol}\n"
        f"Signal: {signal['type']}\n"
        f"Entry: {signal['entry']}\n"
        f"SL: {signal['sl']}\n"
        f"TP: {signal['tp']}\n"
        f"RR: 1:{signal['rr']}\n"
        f"Support: {signal['support']} | Resistance: {signal['resistance']}\n"
        f"Time Stop: {ts_str}\n"
        f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    send_telegram_message(msg)

# ---------- SCAN ----------
def scan_symbol(symbol):
    params = PAIR_PARAMS.get(symbol, {
        'lb_short': DEFAULT_LB_SHORT, 'lb_long': DEFAULT_LB_LONG,
        'adx': DEFAULT_ADX, 'cap': DEFAULT_CAP, 'time_stop': DEFAULT_TIME_STOP
    })
    logging.info(f"Scanning {symbol} (LB={params['lb_short']}/{params['lb_long']}, ADX={params['adx']})")
    
    df4 = fetch_h4_data(symbol, days_back=30)
    if df4 is None or len(df4) < params['lb_long']:
        logging.warning(f"Data tidak cukup untuk {symbol}")
        return

    support, resistance = get_sr(df4, params['lb_short'], params['lb_long'])
    if support.iloc[-1] is np.nan or resistance.iloc[-1] is np.nan:
        logging.warning(f"S/R tidak valid untuk {symbol}")
        return

    ind = calculate_indicators(df4)
    if pd.isna(ind['adx'].iloc[-1]):
        logging.warning(f"Indikator tidak lengkap untuk {symbol}")
        return

    signals = detect_signals(df4, ind, support, resistance, symbol, params)
    if signals:
        for sig in signals:
            if is_duplicate(symbol, sig):
                logging.info(f"{symbol}: Duplicate signal diabaikan")
                continue
            mark_sent(symbol, sig)
            send_alert(symbol, sig)
    else:
        logging.info(f"{symbol}: Tidak ada sinyal")

def scan_all():
    logging.info("=== SCAN DIMULAI ===")
    for sym in SYMBOLS:
        try:
            scan_symbol(sym)
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error {sym}: {e}")
    logging.info("=== SCAN SELESAI ===")

def send_startup():
    msg = (
        f"🤖 **AGEN TRADING AKTIF (FINAL)**\n"
        f"Scan setiap {SCAN_INTERVAL_HOURS} jam\n"
        f"7 Pair Terpilih: {', '.join(SYMBOLS)}"
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
