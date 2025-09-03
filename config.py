import os
from dotenv import load_dotenv
from typing import Optional
import logging

load_dotenv()

class Config:
    """Centralized configuration management"""
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    
    # Trading Configuration
    DEFAULT_SYMBOL = "BTC/USDT"
    CANDLE_TIMEFRAME = "1m"
    SEQUENCE_LENGTH = 180          # Lookback window
    PREDICTION_HORIZON = 1         # Predict next candle
    MAX_ROWS_FOR_LABELING = 10000  # Optional limit for processing
    OUTPUT_DIR = "data"
    
    # Data Collection Configuration (MISSING - Added here)
    FETCH_INTERVAL = 60            # Seconds between fetches
    MAX_RETRIES = 3               # Max retry attempts
    RETRY_DELAY = 5               # Seconds between retries
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return True
import os
from dotenv import load_dotenv
from typing import Optional, List
import logging

load_dotenv()

class Config:
    """Centralized configuration for Zone-Based Trading System (80% Accuracy Target)"""
    
    # Supabase Configuration
    SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_KEY")
    
    # Zone-Based Trading Configuration (PRIMARY SYSTEM)
    DEFAULT_SYMBOL = "BTC/USDT"
    CANDLE_TIMEFRAME = "15m"       # 15-minute candles for zone detection
    SEQUENCE_LENGTH = 96           # 24 hours of 15-min candles (96 candles)
    PREDICTION_HORIZON = 4         # Predict 1 hour ahead (4x15min candles)
    MAX_ROWS_FOR_LABELING = 2000   # Free tier optimization
    OUTPUT_DIR = "data"
    
    # Zone Trading Configuration
    ZONE_CALCULATION_HOURS = 1     # Calculate zones for 1-hour windows
    MIN_CONFIDENCE_THRESHOLD = 0.80  # Only trade 80%+ confidence zones
    ZONE_WIDTH_ATR_MULTIPLIER = 0.5  # Zone width based on ATR
    SUPPORT_RESISTANCE_LOOKBACK = 24  # Hours to look back for S/R levels
    
    # BTC Psychological Levels (important for zones)
    BTC_PSYCHOLOGICAL_INCREMENT = 500  # $500 levels are psychological
    
    # Session Trading Configuration
    TRADING_SESSION_DURATION = 60  # Minutes per trading session
    MAX_SIGNALS_PER_SESSION = 5    # Limit signals to avoid overtrading
    ZONE_RECALCULATION_MINUTES = 30  # Recalculate zones every 30 minutes
    
    # Data Management (Free Tier Optimization)
    DATA_RETENTION_DAYS = 7        # Auto-delete data older than 7 days
    CACHE_DURATION_MINUTES = 15    # Cache predictions for 15 minutes
    FETCH_ONLY_ACTIVE = True       # Only fetch during active hours
    ACTIVE_TRADING_HOURS: List[int] = [9, 10, 11, 12, 13, 14, 15, 16, 1, 2]  # Your trading hours
    
    # API Rate Limiting (Free Tier)
    MAX_API_CALLS_PER_HOUR = 100   # Stay well under Binance limits
    MAX_DB_OPERATIONS_PER_DAY = 1000  # Stay under Supabase limits
    
    # Data Collection Configuration
    FETCH_INTERVAL = 900           # 15 minutes (matching candle timeframe)
    MAX_RETRIES = 3               # Max retry attempts
    RETRY_DELAY = 5               # Seconds between retries
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.SUPABASE_URL or not cls.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return True