import os
import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
import requests
from datetime import datetime

# Token diambil dari environment variable (AMAN, tidak terlihat di kode)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

PAIRS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "EURJPY=X", "EURGBP=X", "NZDUSD=X", "USDCHF=X", "EURCHF=X",
    "EURCAD=X", "GBPJPY=X", "GC=F",
    "AUDJPY=X", "GBPCHF=X", "AUDCAD=X", "NZDJPY=X"
]

ADX_THRESHOLD = 12
VOLUME_MA_PERIOD = 20
VOLUME_RATIO = 0.4
SL_ATR = 1.5
TP_ATR = 2.5
# (SCAN_INTERVAL_HOURS tidak diperlukan lagi, diatur oleh GitHub Actions)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SinyalFinal")

def kirim_telegram(pesan):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": pesan}, timeout=10)
        if resp.status_code != 200:
            logger.warning(f"Telegram gagal: {resp.text}")
        else:
            logger.info("✅ Notifikasi terkirim.")
    except Exception as e:
        logger.error(f"Error: {e}")

def hitung_adx(high, low, close, period=14):
    plus_dm = high.diff().clip(lower=0)
    minus_dm = low.diff().clip(upper=0).abs()
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

def hitung_atr(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def cek_sinyal(symbol):
    logger.info(f"Scanning {symbol}...")
    try:
        df = yf.download(symbol, period="60d", interval="1h", progress=False)
        if df.empty or len(df) < 28:
            return None

        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()
        volume = df['Volume'].squeeze()

        adx, plus_di, minus_di = hitung_adx(high, low, close, 14)
        atr = hitung_atr(high, low, close, 14)
        vol_ma = volume.rolling(VOLUME_MA_PERIOD).mean()

        adx_now = float(adx.iloc[-1])
        plus_now = float(plus_di.iloc[-1])
        minus_now = float(minus_di.iloc[-1])
        atr_now = float(atr.iloc[-1])
        vol_now = float(volume.iloc[-1])
        vol_ma_now = float(vol_ma.iloc[-1])
        price_now = float(close.iloc[-1])

        if pd.isna(adx_now) or pd.isna(atr_now):
            return None

        if vol_now > 0 and not pd.isna(vol_ma_now):
            if vol_now <= VOLUME_RATIO * vol_ma_now:
                logger.info(f"{symbol}: Volume rendah")
                return None

        if adx_now <= ADX_THRESHOLD:
            logger.info(f"{symbol}: ADX rendah ({adx_now:.1f})")
            return None

        if plus_now > minus_now:
            arah = "BUY"
            sl = price_now - (SL_ATR * atr_now)
            tp = price_now + (TP_ATR * atr_now)
        else:
            arah = "SELL"
            sl = price_now + (SL_ATR * atr_now)
            tp = price_now - (TP_ATR * atr_now)

        logger.info(f"🚨 {symbol}: {arah} | Harga: {price_now:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
        return (arah, price_now, sl, tp)

    except Exception as e:
        logger.error(f"Error {symbol}: {e}")
        return None

def main():
    # Cek apakah token tersedia
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_TOKEN atau TELEGRAM_CHAT_ID belum diset di environment.")
        return

    # Kirim notifikasi hanya saat pertama kali workflow berjalan (opsional, bisa dihapus agar tidak spam)
    # kirim_telegram("✅ Bot cloud siap (GitHub Actions).")

    logger.info("=== SCAN DIMULAI ===")
    for pair in PAIRS:
        hasil = cek_sinyal(pair)
        if hasil:
            arah, harga, sl, tp = hasil
            if arah == "BUY":
                emoji = "🟢"
            else:
                emoji = "🔴"
            msg = (f"{emoji} SINYAL {pair}\n"
                   f"Arah: {arah}\n"
                   f"Harga: {harga:.5f}\n"
                   f"Stop Loss: {sl:.5f}\n"
                   f"Take Profit: {tp:.5f}\n"
                   f"Risk/Reward: 1:{TP_ATR/SL_ATR:.1f}")
            kirim_telegram(msg)
        time.sleep(2)

    logger.info("=== SCAN SELESAI ===")

if __name__ == "__main__":
    main()