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
    "GC=F",        # XAUUSD (30 candle, tanpa EMA)
    "USDJPY=X", "GBPJPY=X", "CHFJPY=X", "AUDJPY=X",
    "EURJPY=X", "CADJPY=X", "NZDJPY=X",
    "AUDUSD=X", "EURGBP=X", "NZDUSD=X", "USDCHF=X",
    "EURUSD=X", "GBPUSD=X", "USDCAD=X", "EURNZD=X",
]

PAIR_CONFIG = {
    "GC=F":       {'tp': 500, 'sl': 400, 'pip_value': 0.10, 'swing': 30, 'ema': None},
    "USDJPY=X":   {'tp': 80,  'sl': 75,  'pip_value': 0.01, 'swing': 7, 'ema': None},
    "GBPJPY=X":   {'tp': 150, 'sl': 100, 'pip_value': 0.01, 'swing': 7, 'ema': None},
    "CHFJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01, 'swing': 7, 'ema': None},
    "AUDJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01, 'swing': 7, 'ema': None},
    "EURJPY=X":   {'tp': 150, 'sl': 75,  'pip_value': 0.01, 'swing': 7, 'ema': None},
    "CADJPY=X":   {'tp': 100, 'sl': 40,  'pip_value': 0.01, 'swing': 7, 'ema': None},
    "NZDJPY=X":   {'tp': 100, 'sl': 60,  'pip_value': 0.01, 'swing': 7, 'ema': None},
    "AUDUSD=X":   {'tp': 100, 'sl': 75,  'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "EURGBP=X":   {'tp': 80,  'sl': 75,  'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "NZDUSD=X":   {'tp': 120, 'sl': 100, 'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "USDCHF=X":   {'tp': 80,  'sl': 50,  'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "EURUSD=X":   {'tp': 120, 'sl': 100, 'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "GBPUSD=X":   {'tp': 150, 'sl': 125, 'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "USDCAD=X":   {'tp': 100, 'sl': 60,  'pip_value': 0.0001, 'swing': 7, 'ema': None},
    "EURNZD=X":   {'tp': 120, 'sl': 100, 'pip_value': 0.0001, 'swing': 7, 'ema': None},
}

SENT_LOG_FILE = "sent_signals.json"

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
            df.iat[i, df.columns.get_loc('Top')] = True
        if (l < df['Low'].iloc[i-left:i].values).all() and (l < df['Low'].iloc[i+1:i+1+right].values).all():
            df.iat[i, df.columns.get_loc('Bottom')] = True
    return df

# ================= FETCH DATA H4 =================
def fetch_h4(symbol, days=10):
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1h", progress=False, timeout=60)
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

# ================= STATE DETEKSI DARI DATA =================
def check_existing_position(df, tp_pips, sl_pips, pip_value):
    tp_price = tp_pips * pip_value
    sl_price = sl_pips * pip_value

    last_signal = df[df['Top'] | df['Bottom']].tail(1)
    if last_signal.empty:
        return None

    row = last_signal.iloc[0]
    signal_type = 'Top' if row['Top'] else 'Bottom'
    if signal_type == 'Top':
        entry = row['Low']
        tp = entry - tp_price
        sl = entry + sl_price
        order_type = 'Sell Stop'
    else:
        entry = row['High']
        tp = entry + tp_price
        sl = entry - sl_price
        order_type = 'Buy Stop'

    signal_idx = last_signal.index[0]
    after = df.loc[signal_idx:].iloc[1:]

    if after.empty:
        return None

    triggered = False
    close_outcome = None
    for t, frow in after.iterrows():
        if not triggered:
            if order_type == 'Buy Stop' and frow['High'] >= entry:
                triggered = True
            elif order_type == 'Sell Stop' and frow['Low'] <= entry:
                triggered = True
        if triggered:
            if order_type == 'Buy Stop':
                if frow['High'] >= tp:
                    close_outcome = 'TP'
                    break
                if frow['Low'] <= sl:
                    close_outcome = 'SL'
                    break
            else:
                if frow['Low'] <= tp:
                    close_outcome = 'TP'
                    break
                if frow['High'] >= sl:
                    close_outcome = 'SL'
                    break

    if triggered and close_outcome is None:
        return {'status': 'active', 'type': order_type, 'entry': entry, 'tp': tp, 'sl': sl}
    elif triggered and close_outcome is not None:
        return {'status': 'closed', 'outcome': close_outcome, 'type': order_type, 'entry': entry, 'tp': tp, 'sl': sl}
    else:
        return {'status': 'pending', 'type': order_type, 'entry': entry, 'tp': tp, 'sl': sl}

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
    if df is None or len(df) < 100:
        logging.warning(f"Data tidak cukup {symbol}")
        return

    cfg = PAIR_CONFIG[symbol]
    tp_pips = cfg['tp']
    sl_pips = cfg['sl']
    pip_value = cfg['pip_value']
    swing_n = cfg['swing']
    ema_period = cfg['ema']

    # Deteksi swing sesuai aturan
    if swing_n == 30:
        df = detect_swings(df, left=15, right=15)
    elif swing_n == 7:
        df = detect_swings(df, left=3, right=3)
    else:
        # fallback untuk aturan lain (misal 11 candle)
        df = detect_swings(df, left=5, right=5)

    # Filter EMA hanya jika diaktifkan
    if ema_period is not None:
        df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
        df = df[(df['Top'] & (df['Close'] < df['EMA'])) | (df['Bottom'] & (df['Close'] > df['EMA']))]

    # --- Stateful Check ---
    pos = check_existing_position(df, tp_pips, sl_pips, pip_value)
    if pos:
        if pos['status'] == 'active':
            logging.info(f"{symbol}: Posisi masih aktif ({pos['type']}). Sinyal baru diabaikan.")
            return
        elif pos['status'] == 'pending':
            logging.info(f"{symbol}: Pending order masih berlaku ({pos['type']}). Sinyal baru diabaikan.")
            return
        elif pos['status'] == 'closed':
            msg = (
                f"✅ **POSISI CLOSE**\n"
                f"Symbol: {symbol}\n"
                f"Order: {pos['type']}\n"
                f"Entry: {pos['entry']:.5f}\n"
                f"TP: {pos['tp']:.5f} ({tp_pips} pip)\n"
                f"SL: {pos['sl']:.5f} ({sl_pips} pip)\n"
                f"Close: {pos['outcome']} {'✅' if pos['outcome'] == 'TP' else '❌'}\n"
                f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            send_telegram(msg)
            logging.info(f"{symbol}: Posisi closed ({pos['outcome']}). Mengizinkan sinyal baru.")

    # --- Deteksi sinyal baru ---
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
        f"Filter: {'EMA ' + str(ema_period) if ema_period else 'Tanpa Filter'}\n"
        f"Candle close: {candle_end_time}\n"
        f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    if send_telegram(msg):
        mark_signal_sent(symbol, signal_type, timestamp_str, sent_log)
        logging.info(f"{symbol}: Sinyal {order_type} dikirim.")

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
            time.sleep(1)
        except Exception as e:
            logging.error(f"Error {sym}: {e}")
    elapsed = time.time() - start_time
    logging.info(f"=== SCAN SELESAI dalam {elapsed:.2f} detik ===")

if __name__ == "__main__":
    main()
