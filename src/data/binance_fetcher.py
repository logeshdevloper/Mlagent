import ccxt
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List
from config import Config
from src.api.supabase_client import SupabaseClient
from src.utils.logger import setup_logger
import pandas as pd

class BinanceFetcher:
    """Optimized Binance data fetcher for zone-based trading"""

    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True
            }
        })
        self.logger = setup_logger(__name__, "fetcher.log")
        self.db_client = SupabaseClient()
        self.candle_cache = {}  # Cache for reducing API calls
        self.last_fetch_time = {}
    
    def validate_symbol(self, symbol: str) -> bool:
        """Ensure the trading pair is valid on Binance"""
        markets = self.exchange.load_markets()
        if symbol not in markets:
            self.logger.error(f"Symbol {symbol} not available on Binance.")
            return False
        return True

    def fetch_latest_candle(self, symbol: str = Config.DEFAULT_SYMBOL, timeframe: str = None) -> Optional[dict[str, any]]:
        """Fetch the latest completed candle with full error resilience"""
        try:
            if not self.validate_symbol(symbol):
                return None

            timeframe = timeframe or Config.CANDLE_TIMEFRAME
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)
            if len(ohlcv) < 2:
                self.logger.warning(f"Not enough candles returned for {symbol}")
                return None
            
            candle = ohlcv[-2]
            candle_data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(candle[0] / 1000, tz=timezone.utc).isoformat(),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5])
            }
            return candle_data

        except ccxt.RequestTimeout as e:
            self.logger.warning(f"Timeout when fetching from Binance: {e}")
            return None
        except ccxt.NetworkError as e:
            self.logger.warning(f"Network issue: {e}")
            return None
        except ccxt.ExchangeError as e:
            self.logger.error(f"Exchange error from Binance: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None
        

    def fetch_multi_timeframe_data(self, symbol: str = Config.DEFAULT_SYMBOL) -> Dict:
        """Fetch data from multiple timeframes for better zone detection"""
        try:
            data = {}
            timeframes = ['5m', '15m', '1h']
            
            for tf in timeframes:
                # Check cache first
                cache_key = f"{symbol}_{tf}"
                if cache_key in self.candle_cache:
                    cache_time = self.last_fetch_time.get(cache_key, 0)
                    if time.time() - cache_time < 300:  # 5-minute cache
                        data[tf] = self.candle_cache[cache_key]
                        continue
                
                # Fetch from exchange
                candles = self.fetch_candles_for_timeframe(symbol, tf, limit=100)
                if candles:
                    data[tf] = candles
                    self.candle_cache[cache_key] = candles
                    self.last_fetch_time[cache_key] = time.time()
            
            return data
        except Exception as e:
            self.logger.error(f"Multi-timeframe fetch failed: {e}")
            return {}
    
    def fetch_candles_for_timeframe(self, symbol: str, timeframe: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch multiple candles for a specific timeframe"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.logger.error(f"Failed to fetch {timeframe} candles: {e}")
            return None
    
    def fetch_session_data(self, symbol: str = Config.DEFAULT_SYMBOL) -> Dict:
        """Fetch data optimized for trading session (reduced API calls)"""
        try:
            # Only fetch if we're in active trading hours
            current_hour = datetime.now().hour
            if Config.FETCH_ONLY_ACTIVE and current_hour not in Config.ACTIVE_TRADING_HOURS:
                self.logger.info(f"Outside active trading hours ({current_hour}). Skipping fetch.")
                return {}
            
            # Fetch last 24 hours of 15-min candles (96 candles)
            df = self.fetch_candles_for_timeframe(symbol, Config.CANDLE_TIMEFRAME, limit=96)
            
            if df is not None and not df.empty:
                return {
                    'symbol': symbol,
                    'timeframe': Config.CANDLE_TIMEFRAME,
                    'data': df,
                    'current_price': float(df['close'].iloc[-1]),
                    'timestamp': datetime.now().isoformat()
                }
            
            return {}
        except Exception as e:
            self.logger.error(f"Session data fetch failed: {e}")
            return {}
    
    def run_collector(self, symbol: str = Config.DEFAULT_SYMBOL, max_iterations: Optional[int] = None):
        """Optimized data collection for zone-based trading"""
        self.logger.info(f"Starting optimized data collection for {symbol}")
        
        iteration_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        try:
            while True:
                # Check iteration limit
                if max_iterations and iteration_count >= max_iterations:
                    self.logger.info(f"Reached maximum iterations: {max_iterations}")
                    break
                
                start_time = time.time()
                
                # Check if we're in active trading hours
                current_hour = datetime.now().hour
                if Config.FETCH_ONLY_ACTIVE and current_hour not in Config.ACTIVE_TRADING_HOURS:
                    self.logger.info(f"Outside active hours. Sleeping for 1 hour...")
                    time.sleep(3600)  # Sleep for 1 hour
                    continue
                
                # Fetch and store candle (15-min timeframe)
                candle_data = self.fetch_latest_candle(symbol)
                
                if candle_data:
                    success = self.db_client.insert_candle(candle_data)
                    
                    if success:
                        consecutive_failures = 0
                        total_candles = self.db_client.get_candle_count(symbol)
                        self.logger.info(
                            f"[{datetime.now()}] Stored candle for {symbol} "
                            f"(Close: {candle_data['close']}, Total: {total_candles})"
                        )
                    else:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1
                
                # Handle consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error(
                        f"Too many consecutive failures ({consecutive_failures}). "
                        "Stopping collector."
                    )
                    break
                
                # Calculate sleep time to maintain interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, Config.FETCH_INTERVAL - elapsed_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                iteration_count += 1
                
        except KeyboardInterrupt:
            self.logger.info("Data collection stopped by user")
        except Exception as e:
            self.logger.error(f"Unexpected error in collector: {e}")
            raise

