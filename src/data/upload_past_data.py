# scripts/fetch_and_upload_history.py

import ccxt
from datetime import datetime, timedelta, timezone
import time
from src.api.supabase_client import SupabaseClient
from src.utils.logger import setup_logger

logger = setup_logger("historical_fetcher")

def fetch_binance_historical(symbol: str, days_back: int = 7, timeframe: str = "15m") -> list[dict]:
    """
    Fetch historical 15-min candles for the past N days for a symbol from Binance.
    Returns: list of dicts ready to be inserted into Supabase
    """
    logger.info(f"📦 Fetching historical {timeframe} candles for {symbol} | Last {days_back} days")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })

    since_ms = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
    all_candles = []
    max_limit = 1000
    # For 15-min candles: 4 per hour * 24 hours * days = 96 candles per day
    total_candles_expected = days_back * 96
    max_attempts = (total_candles_expected // max_limit) + 1

    for attempt in range(max_attempts):
        logger.info(f"⏳ Fetching batch {attempt+1}")
        candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=max_limit)

        if not candles:
            logger.warning("⚠️ No candles returned — breaking")
            break

        for candle in candles:
            all_candles.append({
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc).isoformat(),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            })

        since_ms = candles[-1][0] + 900_000  # move forward by 15 min (900,000 ms)
        time.sleep(0.5)  # avoid hitting rate limits

    logger.info(f"✅ Total candles fetched: {len(all_candles)}")
    return all_candles


def upload_to_supabase(candles: list[dict], batch_size: int = 500):
    """
    Upload candle data to Supabase in batches
    """
    logger.info("🚀 Uploading candles to Supabase...")
    client = SupabaseClient()

    total = len(candles)
    for i in range(0, total, batch_size):
        batch = candles[i:i + batch_size]
        try:
            res = client.client.table("candles").insert(batch).execute()
            logger.info(f"✅ Uploaded batch {i} → {i+len(batch)}")
        except Exception as e:
            logger.error(f"❌ Failed batch {i} → {i+len(batch)}: {e}")

    logger.info(f"🎯 Done uploading {total} candles.")
