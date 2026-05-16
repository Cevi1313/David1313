import os
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import json
import hashlib
from datetime import datetime, timedelta, timezone
from telegram import Bot
from telegram.error import TelegramError, TimedOut, RetryAfter

# ================= KONFIGURASI =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = [
    "GC=F",        # XAUUSD
    "USDJPY=X",
    "GBPJPY=X",
    "CHFJPY=X",
    "AUDJPY=X",
    "EURJPY=X",
    "CADJPY=X",
    "NZDJPY=X",
]

PAIR_CONFIG = {
    "GC=F":       {'tp': 500, 'sl': 300, 'pip_value': 0.10},
    "USDJPY=X":   {'tp': 80,  'sl': 75,  'pip_value': 0.01},
    "GBPJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01},
    "CHFJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01},
    "AUDJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01},
    "EURJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01},
    "CADJPY=X":   {'tp': 100, 'sl': 40,  'pip_value': 0.01},
    "NZDJPY=X":   {'tp': 100, 'sl': 60,  'pip_value': 0.01},
}

STATE_FILE = "trading_state.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
bot = Bot(token=TELEGRAM_TOKEN)

# ================= FUNGSI DETEKSI SWING =================
def detect_swings(df, left=3, right=3):
    df = df.copy()
    df['Top'] = False
    df['Bottom'] = False
    for i in range(left, len(df) - right):
        h = df['High'].iloc[i]
        l = df['Low'].iloc[i]
        if (h > df['High'].iloc[i-left:i].values).all() and (h > df['High'].iloc[i+1:i+1+right].values).all():
            df.at[df.index[i], 'Top'] = True
        if (l < df['Low'].iloc[i-left:i].values).all() and (l < df['Low'].iloc[i+1:i+1+right].values).all():
            df.at[df.index[i], 'Bottom'] = True
    return df

# ================= FETCH DATA H4 =================
def fetch_h4(symbol):
    try:
        df = yf.download(symbol, period="5d", interval="1h", progress=False, timeout=30)
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

# ================= STATE FILE =================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

# ================= ANTI DUPLIKAT SIGNAL =================
def generate_signal_hash(symbol, sig_type, timestamp):
    raw = f"{symbol}_{sig_type}_{timestamp}"
    return hashlib.md5(raw.encode()).hexdigest()

def is_duplicate(symbol, sig_type, timestamp, state):
    key = generate_signal_hash(symbol, sig_type, timestamp)
    last = state.get(key)
    if last:
        last_time = datetime.fromisoformat(last)
        if datetime.now() - last_time < timedelta(hours=24):
            return True
    return False

def mark_sent(symbol, sig_type, timestamp, state):
    key = generate_signal_hash(symbol, sig_type, timestamp)
    state[key] = datetime.now().isoformat()
    save_state(state)

# ================= CLEAR POSISI OTOMATIS =================
def check_and_clear_position(symbol, df, state):
    pos = state.get(f"position_{symbol}")
    if not pos:
        return None

    signal_time = pd.Timestamp(pos['signal_time'])
    mask = df.index > signal_time
    if not mask.any():
        return pos

    after_df = df.loc[mask]
    entry = pos['entry']
    tp = pos['tp']
    sl = pos['sl']
    order_type = pos['type']

    if order_type == 'Buy Stop':
        hit_tp = (after_df['High'] >= tp).any()
        hit_sl = (after_df['Low'] <= sl).any()
    else:
        hit_tp = (after_df['Low'] <= tp).any()
        hit_sl = (after_df['High'] >= sl).any()

    if hit_tp or hit_sl:
        outcome = 'TP' if hit_tp else 'SL'
        msg = (
            f"✅ **POSISI CLOSE**\n"
            f"Symbol: {symbol}\n"
            f"Order: {order_type}\n"
            f"Entry: {entry:.5f}\n"
            f"Close: {outcome} {'✅' if outcome == 'TP' else '❌'}\n"
            f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        send_telegram(msg)
        del state[f"position_{symbol}"]
        save_state(state)
        logging.info(f"{symbol}: Posisi {order_type} closed ({outcome}) dan dihapus.")
        return None
    else:
        return pos

def save_position(symbol, pos_data, state):
    state[f"position_{symbol}"] = pos_data
    save_state(state)

# ================= KIRIM TELEGRAM =================
def send_telegram(msg, max_retries=3):
    for attempt in range(max_retries):
        try:
            bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg, timeout=10)
            logging.info("✅ Notifikasi terkirim")
            return True
        except (TimedOut, RetryAfter, TelegramError) as e:
            wait = 2 ** attempt
            logging.warning(f"Telegram error, retry {attempt+1}/{max_retries}: {e}")
            time.sleep(wait)
        except Exception as e:
            logging.error(f"Error umum telegram: {e}")
            break
    logging.error("❌ Gagal kirim pesan ke Telegram")
    return False

# ================= MAIN SCAN =================
def scan_symbol(symbol):
    logging.info(f"Scanning {symbol}...")
    df = fetch_h4(symbol)
    if df is None or len(df) < 100:
        logging.warning(f"Data tidak cukup {symbol}")
        return

    state = load_state()

    open_pos = check_and_clear_position(symbol, df, state)
    if open_pos:
        logging.info(f"{symbol}: Posisi masih berjalan ({open_pos['type']}). Sinyal baru diabaikan.")
        return

    cfg = PAIR_CONFIG.get(symbol, PAIR_CONFIG["GC=F"])
    tp_pips = cfg['tp']
    sl_pips = cfg['sl']
    pip_value = cfg['pip_value']

    df = detect_swings(df)

    last_candle = df.iloc[-1]
    candle_end_time = last_candle.name
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    if now_utc < candle_end_time:
        logging.info(f"Candle {symbol} belum selesai, tunda.")
        return

    if not (last_candle['Top'] or last_candle['Bottom']):
        logging.info(f"{symbol}: Tidak ada sinyal.")
        return

    signal_type = 'Top' if last_candle['Top'] else 'Bottom'
    timestamp_str = str(candle_end_time)

    if is_duplicate(symbol, signal_type, timestamp_str, state):
        logging.info(f"{symbol}: Sinyal sudah dikirim.")
        return

    if signal_type == 'Top':
        entry = last_candle['Low']
        tp = entry - tp_pips * pip_value
        sl = entry + sl_pips * pip_value
        order_type = 'Sell Stop'
    else:
        entry = last_candle['High']
        tp = entry + tp_pips * pip_value
        sl = entry - sl_pips * pip_value
        order_type = 'Buy Stop'

    msg = (
        f"🔔 **SINYAL REVERSAL**\n"
        f"Symbol: {symbol}\n"
        f"Signal: {signal_type}\n"
        f"Order: {order_type} di {entry:.5f}\n"
        f"TP: {tp:.5f} ({tp_pips} pip)\n"
        f"SL: {sl:.5f} ({sl_pips} pip)\n"
        f"Candle close: {candle_end_time}\n"
        f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if send_telegram(msg):
        mark_sent(symbol, signal_type, timestamp_str, state)
        save_position(symbol, {
            'type': order_type,
            'entry': entry,
            'tp': tp,
            'sl': sl,
            'signal_time': timestamp_str
        }, state)
        logging.info(f"{symbol}: {order_type} terpasang di {entry:.5f}")

def main():
    # Cek hari – jangan jalankan di Sabtu/Minggu (UTC)
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()
    if weekday >= 5:
        logging.info("Hari Sabtu/Minggu – pasar libur, scan dihentikan.")
        return

    logging.info("=== SCAN DIMULAI ===")
    start_time = time.time()
    for sym in SYMBOLS:
        try:
            scan_symbol(sym)
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error {sym}: {e}")
    elapsed = time.time() - start_time
    logging.info(f"=== SCAN SELESAI dalam {elapsed:.2f} detik ===")

if __name__ == "__main__":
    main()
