# ==========================================================
# FILE: telegram_delta_scanner.py
# DESCRIPTION: Delta Exchange Scanner (1H Trend + 15M Confluence)
# ENVIRONMENT: GitHub Actions Ready (Single-Pass Parallel Execution)
# ==========================================================

import os
import sys
import time
import threading
import requests
import pandas as pd
import numpy as np
import ccxt
from concurrent.futures import ThreadPoolExecutor

# Ensure UTF-8 output encoding for terminal/runner printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================================
# 1. Credentials & Exchange Setup
# ==========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# CCXT Exchange Setup (Delta Exchange Public Market Data)
exchange = ccxt.delta({'enableRateLimit': True})

# Guaranteed Priority Symbols (Always scanned first)
CORE_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'XAU/USDT', 'XAG/USDT']

# Thread safety lock for shared alert state
state_lock = threading.Lock()
last_alerted_candle = {}


def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print(f"[Alert Output (Telegram credentials not set)]:\n{message}\n")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'Markdown'
    }
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code != 200:
            print(f"Telegram Alert API Response Error ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"Telegram Alert Error: {e}")


def get_dynamic_delta_watchlist():
    """
    Fetch top 50 active perpetual contracts from Delta Exchange.
    Guarantees Priority Assets (BTC, ETH, SOL, XRP, Gold, Silver) are ALWAYS included,
    along with top liquid US Stocks and cryptos.
    """
    watchlist = list(CORE_SYMBOLS)
    try:
        # 1. Load CCXT Markets to verify supported pair formats
        markets = exchange.load_markets()
        market_symbols = list(markets.keys())

        # 2. Fetch live products from Delta REST API
        url = "https://api.india.delta.exchange/v2/products"
        res = requests.get(url, timeout=10).json()

        if res.get("success") and "result" in res:
            for p in res["result"]:
                sym = p.get("symbol", "")
                c_type = p.get("contract_type", "")
                state = p.get("state", "")

                # Only live perpetual contracts (Cryptos, Synthetic US Stocks & Commodities)
                if state == "live" and "perpetual" in c_type:
                    if sym in ["USDCUSDT", "USDC/USDT", "USD_USDT"]:
                        continue

                    # Standardize CCXT symbol notation
                    formatted_sym = None
                    if sym in market_symbols:
                        formatted_sym = sym
                    elif "/" not in sym:
                        if sym.endswith("USDT") and f"{sym[:-4]}/USDT" in market_symbols:
                            formatted_sym = f"{sym[:-4]}/USDT"
                        elif sym.endswith("USD") and f"{sym[:-3]}/USD:USD" in market_symbols:
                            formatted_sym = f"{sym[:-3]}/USD:USD"
                        elif sym.endswith("USD") and f"{sym[:-3]}/USD" in market_symbols:
                            formatted_sym = f"{sym[:-3]}/USD"

                    if not formatted_sym:
                        if "/" not in sym and sym.endswith("USDT"):
                            formatted_sym = f"{sym[:-4]}/USDT"
                        else:
                            formatted_sym = sym

                    if formatted_sym not in watchlist:
                        watchlist.append(formatted_sym)

        # टॉप 50 सिम्बल्स तक सीमित करें
        watchlist = watchlist[:50]
        print(f"[+] Loaded {len(watchlist)} total assets (Priority + Cryptos + US Stocks + Commodities).")
        return watchlist
    except Exception as e:
        print(f"[-] Error loading dynamic watchlist: {e}. Defaulting to CORE_SYMBOLS.")
        return CORE_SYMBOLS


def fetch_ohlcv(symbol, timeframe, limit=250):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) == 0:
            return None
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        return df
    except Exception:
        return None


def compute_indicators(df_15m, df_1h):
    # --- 1H HTF Filters (200 EMA + ADX) ---
    df_1h['ema_200'] = df_1h['close'].ewm(span=200, adjust=False).mean()
    
    h_1h, l_1h, c_1h = df_1h['high'], df_1h['low'], df_1h['close']
    tr_1h = pd.concat([
        h_1h - l_1h,
        (h_1h - c_1h.shift(1)).abs(),
        (l_1h - c_1h.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_1h = tr_1h.rolling(14).mean()
    
    up_move = h_1h - h_1h.shift(1)
    down_move = l_1h.shift(1) - l_1h
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * (pd.Series(plus_dm, index=df_1h.index).rolling(14).mean() / (atr_1h + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df_1h.index).rolling(14).mean() / (atr_1h + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df_1h['adx'] = dx.rolling(14).mean()
    
    # --- 15M Execution Indicators ---
    delta = df_15m['close'].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df_15m['rsi'] = 100 - (100 / (1 + rs))
    
    tr_15 = pd.concat([
        df_15m['high'] - df_15m['low'],
        (df_15m['high'] - df_15m['close'].shift(1)).abs(),
        (df_15m['low'] - df_15m['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df_15m['atr'] = tr_15.rolling(14).mean()
    df_15m['vol_ma'] = df_15m['volume'].rolling(20).mean()
    
    latest_1h_ema = df_1h['ema_200'].iloc[-1]
    latest_1h_adx = df_1h['adx'].iloc[-1]
    
    return df_15m, latest_1h_ema, latest_1h_adx


def scan_symbol(symbol):
    df_15m = fetch_ohlcv(symbol, '15m', limit=60)
    df_1h = fetch_ohlcv(symbol, '1h', limit=250)
    
    if df_15m is None or df_1h is None or len(df_15m) < 30 or len(df_1h) < 205:
        return
        
    df_15m, htf_ema, htf_adx = compute_indicators(df_15m, df_1h)
    closed_candle_time = df_15m.index[-2]
    
    with state_lock:
        if last_alerted_candle.get(symbol) == closed_candle_time:
            return
        
    c_prev2 = df_15m['close'].iloc[-3]
    c_prev1 = df_15m['close'].iloc[-2]
    rsi_prev2 = df_15m['rsi'].iloc[-3]
    rsi_prev1 = df_15m['rsi'].iloc[-2]
    vol_prev1 = df_15m['volume'].iloc[-2]
    vol_ma_prev1 = df_15m['vol_ma'].iloc[-2]
    atr_val = df_15m['atr'].iloc[-2]
    
    # 1. BUY TRIGGER
    if (c_prev1 > htf_ema) and (htf_adx > 20):
        if (rsi_prev2 <= 45 and rsi_prev1 > 45) and (vol_prev1 > 1.2 * vol_ma_prev1):
            sl_dist = 1.5 * atr_val
            tp_dist = 4.5 * atr_val
            sl_price = c_prev1 - sl_dist
            tp_price = c_prev1 + tp_dist
            
            alert = (
                f"🟢 *BUY SIGNAL (1:3 RR)*\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• *Asset*: `{symbol}` (15M)\n"
                f"• *Entry (Close)*: `{c_prev1:.4f}`\n"
                f"• *Stop-Loss*: `{sl_price:.4f}` (-{sl_dist:.4f})\n"
                f"• *Take-Profit*: `{tp_price:.4f}` (+{tp_dist:.4f})\n"
                f"• *1H Context*: Bullish | ADX: `{htf_adx:.1f}`\n"
                f"• *Volume*: `{vol_prev1 / (vol_ma_prev1 + 1e-9):.2f}x` 20-MA\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ *Check*: Support / Liquidity Sweep Confirmation?\n"
                f"📌 *Action*: Place Limit Maker Order!"
            )
            print(f"[!] BUY Signal sent for {symbol} at {closed_candle_time}")
            send_telegram_alert(alert)
            with state_lock:
                last_alerted_candle[symbol] = closed_candle_time

    # 2. SELL TRIGGER
    elif (c_prev1 < htf_ema) and (htf_adx > 20):
        if (rsi_prev2 >= 55 and rsi_prev1 < 55) and (vol_prev1 > 1.2 * vol_ma_prev1):
            sl_dist = 1.5 * atr_val
            tp_dist = 4.5 * atr_val
            sl_price = c_prev1 + sl_dist
            tp_price = c_prev1 - tp_dist
            
            alert = (
                f"🔴 *SELL SIGNAL (1:3 RR)*\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• *Asset*: `{symbol}` (15M)\n"
                f"• *Entry (Close)*: `{c_prev1:.4f}`\n"
                f"• *Stop-Loss*: `{sl_price:.4f}` (+{sl_dist:.4f})\n"
                f"• *Take-Profit*: `{tp_price:.4f}` (-{tp_dist:.4f})\n"
                f"• *1H Context*: Bearish | ADX: `{htf_adx:.1f}`\n"
                f"• *Volume*: `{vol_prev1 / (vol_ma_prev1 + 1e-9):.2f}x` 20-MA\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"⚠️ *Check*: Resistance / Liquidity Sweep Confirmation?\n"
                f"📌 *Action*: Place Limit Maker Order!"
            )
            print(f"[!] SELL Signal sent for {symbol} at {closed_candle_time}")
            send_telegram_alert(alert)
            with state_lock:
                last_alerted_candle[symbol] = closed_candle_time


def safe_scan_symbol(symbol):
    """Safe wrapper so individual symbol errors do not disrupt parallel execution"""
    try:
        scan_symbol(symbol)
    except Exception as e:
        print(f"Error scanning {symbol}: {e}")


def main():
    print("[+] Starting Delta Exchange Scanner (Single-pass, GitHub Actions Ready)...")
    cycle_start_time = time.time()
    
    current_watchlist = get_dynamic_delta_watchlist()
    
    # Parallelize symbol scanning across 8 worker threads
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(safe_scan_symbol, current_watchlist))
        
    scan_duration = time.time() - cycle_start_time
    print(f"[+] Single-pass scan completed for {len(current_watchlist)} symbols in {scan_duration:.2f} seconds.")


main() 
