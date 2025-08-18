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