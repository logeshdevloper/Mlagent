import ccxt
import time
from datetime import datetime, timezone
from typing import Optional
from config import Config
from src.api.supabase_client import SupabaseClient
from src.utils.logger import setup_logger

class BinanceFetcher:
    """Robust Binance data fetcher"""

    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True
            }
        })
        self.logger = setup_logger(__name__, "fetcher.log")
        self.db_client = SupabaseClient()
    
    def validate_symbol(self, symbol: str) -> bool:
        """Ensure the trading pair is valid on Binance"""
        markets = self.exchange.load_markets()
        if symbol not in markets:
            self.logger.error(f"Symbol {symbol} not available on Binance.")
            return False
        return True

    def fetch_latest_candle(self, symbol: str = Config.DEFAULT_SYMBOL) -> Optional[dict[str, any]]:
        """Fetch the latest completed candle with full error resilience"""
        try:
            if not self.validate_symbol(symbol):
                return None

            ohlcv = self.exchange.fetch_ohlcv(symbol, Config.CANDLE_TIMEFRAME, limit=2)
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
        

    def run_collector(self, symbol: str = Config.DEFAULT_SYMBOL, max_iterations: Optional[int] = None):
        """Run the continuous data collection loop"""
        self.logger.info(f"Starting data collection for {symbol}")
        
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
                
                # Fetch and store candle
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

