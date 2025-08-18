from supabase import create_client, Client
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
from config import Config
from src.utils.logger import setup_logger

class SupabaseClient:
    """Supabase database client with error handling"""
    
    def __init__(self):
        Config.validate()
        self.client: Client = create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
        self.logger = setup_logger(__name__, "supabase.log")
        
    def insert_candle(self, candle_data: Dict[str, Any]) -> bool:
        """Insert a single candle into the database"""
        try:
            response = self.client.table("candles").insert([candle_data]).execute()
            self.logger.debug(f"Inserted candle: {candle_data['symbol']} at {candle_data['timestamp']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to insert candle: {e}")
            return False
    
    def get_candles(self, symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Retrieve candles from database"""
        try:
            response = (self.client.table("candles")
                       .select("*")
                       .eq("symbol", symbol)
                       .order("timestamp", desc=False)
                       .limit(limit)
                       .execute())
            
            if not response.data:
                self.logger.warning(f"No candles found for {symbol}")
                return None
                
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            self.logger.info(f"Retrieved {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve candles: {e}")
            return None
    
    def get_candle_count(self, symbol: str) -> int:
        """Get total candle count for a symbol"""
        try:
            response = (self.client.table("candles")
                       .select("id", count="exact")
                       .eq("symbol", symbol)
                       .execute())
            return response.count or 0
            
        except Exception as e:
            self.logger.error(f"Failed to get candle count: {e}")
            return 0
    
    def create_table_if_not_exists(self) -> bool:
        """Create candles table if it doesn't exist"""
        try:
            # Note: This requires SUPABASE_KEY to have sufficient privileges
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS candles (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(symbol, timestamp)
            );
            
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_timestamp 
            ON candles(symbol, timestamp);
            """
            
            # This would typically be run via Supabase SQL editor
            self.logger.info("Table creation SQL ready - run via Supabase dashboard")
            return True
            
        except Exception as e:
            self.logger.error(f"Table creation guidance failed: {e}")
            return False
