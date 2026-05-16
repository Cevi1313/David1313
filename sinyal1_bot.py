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
    "GC=F",
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

SENT_LOG_FILE = "sent_signals.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
bot = Bot(token=TELEGRAM_TOKEN)

# ================= DETEKSI SWING =================
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

# ================= FETCH H4 (ringan, 2 hari) =================
def fetch_h4(symbol):
    try:
        df = yf.download(symbol, period="2d", interval="1h", progress=False, timeout=30)
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

# ================= LOG SINYAL TERKIRIM =================
def load_sent_log():
    if os.path.exists(SENT_LOG_FILE):
        with open(SENT_LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_sent_log(log):
    with open(SENT_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def signal_already_sent(symbol, sig_type, timestamp, log):
    key = f"{symbol}_{sig_type}_{timestamp}"
    return key in log

def mark_signal_sent(symbol, sig_type, timestamp, log):
    key = f"{symbol}_{sig_type}_{timestamp}"
    log[key] = datetime.now().isoformat()
    save_sent_log(log)

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
    if df is None or len(df) < 7:   # minimal 7 candle (3 kiri + sinyal + 3 kanan)
        logging.warning(f"Data tidak cukup {symbol}")
        return

    cfg = PAIR_CONFIG.get(symbol, PAIR_CONFIG["GC=F"])
    tp_pips = cfg['tp']
    sl_pips = cfg['sl']
    pip_value = cfg['pip_value']

    df = detect_swings(df)

    # Ambil candle terakhir yang sudah selesai (harusnya indeks ke-2 dari akhir)
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

    sent_log = load_sent_log()
    if signal_already_sent(symbol, signal_type, timestamp_str, sent_log):
        logging.info(f"{symbol}: Sinyal sudah dikirim sebelumnya.")
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
        mark_signal_sent(symbol, signal_type, timestamp_str, sent_log)
        logging.info(f"{symbol}: Sinyal {order_type} dikirim.")
    else:
        logging.warning(f"{symbol}: Gagal mengirim sinyal.")

def main():
    now_utc = datetime.now(timezone.utc)
    if now_utc.weekday() >= 5:
        logging.info("Hari Sabtu/Minggu – pasar libur, scan dihentikan.")
        return

    logging.info("=== SCAN DIMULAI ===")
    start_time = time.time()
    for sym in SYMBOLS:
        try:
            scan_symbol(sym)
            time.sleep(1)   # lebih cepat
        except Exception as e:
            logging.error(f"Error {sym}: {e}")
    elapsed = time.time() - start_time
    logging.info(f"=== SCAN SELESAI dalam {elapsed:.2f} detik ===")

if __name__ == "__main__":
    main()
