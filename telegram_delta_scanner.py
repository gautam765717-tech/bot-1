# ==========================================================
# FILE: telegram_delta_scanner.py
# DESCRIPTION: Dynamic Top Multi-Symbol Scanner (1H Trend + 15M Confluence)
# OPTIMIZED: Parallelized Data Fetching with 8 Workers
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

# Ensure UTF-8 output encoding for terminal printing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ==========================================================
# 1. Credentials & Setup
# ==========================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8806015211:AAFZ0869Pel2B8Ho5woriygKPI_qUQPgps0")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1004300378073")

# CCXT Exchange Setup (Public Market Data)
exchange = ccxt.binance({
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# Pure Crypto USDT Core Symbols
CORE_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 
    'BNB/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT',
    'LINK/USDT', 'DOT/USDT'
]

# Thread safety lock for shared alert state
state_lock = threading.Lock()
last_alerted_candle = {}


def send_telegram_alert(message):
    if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE" or TELEGRAM_CHAT_ID == "YOUR_CHAT_ID_HERE":
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


def get_dynamic_top_10_watchlist():
    """Fetch live 24h volume from exchange to dynamically build top watchlist of pure Crypto USDT pairs"""
    try:
        tickers = exchange.fetch_tickers()
        volume_list = []
        
        # Exclude non-crypto commodities (gold/silver), equities/stocks, and leverage tokens
        stock_commodity_bases = {
            'XAU', 'XAG', 'GOLD', 'SILVER', 'NVDA', 'MSTR', 'TSLA', 'AAPL', 
            'AMZN', 'GOOG', 'MSFT', 'META', 'MU', 'SOXL', 'SKHYNIX', 'SKHY', 
            'SPCX', 'KORU', 'SNDK', 'SNXX'
        }
        invalid_keywords = ['UP/', 'DOWN/', 'BEAR/', 'BULL/']
        
        for symbol, data in tickers.items():
            if '/USDT' in symbol and data.get('quoteVolume') is not None:
                clean_symbol = symbol.split(':')[0].strip()
                if not clean_symbol or clean_symbol.startswith('/'):
                    continue
                
                parts = clean_symbol.split('/')
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    continue
                
                base_symbol = parts[0].upper()
                quote_symbol = parts[1].upper()
                
                # Base asset must be a valid crypto symbol (ASCII alphanumeric, not identical to quote, at least 2 chars)
                if not base_symbol.isascii() or base_symbol == quote_symbol or len(base_symbol) < 2 or not base_symbol.isalnum():
                    continue
                if base_symbol in stock_commodity_bases:
                    continue
                if any(kw in clean_symbol.upper() for kw in invalid_keywords):
                    continue
                
                volume_list.append({
                    'symbol': clean_symbol,
                    'volume': data['quoteVolume']
                })
        
        df_vol = pd.DataFrame(volume_list).drop_duplicates(subset=['symbol']).sort_values(by='volume', ascending=False)
        top_vol_symbols = df_vol['symbol'].tolist()
        
        final_list = list(CORE_SYMBOLS)
        for sym in top_vol_symbols:
            if len(final_list) >= 50:
                break
            if sym not in final_list:
                final_list.append(sym)
                
        safe_list_str = str(final_list).encode('ascii', errors='ignore').decode('ascii')
        print(f"[+] Updated Top 50 Watchlist ({len(final_list)} pure crypto symbols): {safe_list_str}")
        return final_list
    except Exception as e:
        print(f"Error updating watchlist: {e}")
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
    except Exception as e:
        print(f"Error fetching {symbol} {timeframe}: {e}")
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
    print("[+] Starting Parallelized Dynamic Top 50 Scanner Bot (8 Thread Workers)...")
    current_watchlist = get_dynamic_top_10_watchlist()
    last_watchlist_update = time.time()
    
    while True:
        try:
            cycle_start_time = time.time()
            
            if time.time() - last_watchlist_update > 3600:
                current_watchlist = get_dynamic_top_10_watchlist()
                last_watchlist_update = time.time()
                
            # Parallelize symbol scanning across 8 worker threads
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(safe_scan_symbol, current_watchlist))
                
            scan_duration = time.time() - cycle_start_time
            print(f"[+] Completed scanning {len(current_watchlist)} symbols in {scan_duration:.2f} seconds.")
            
            time.sleep(30)
        except Exception as e:
            print(f"Loop Error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
