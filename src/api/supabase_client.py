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
    
    def insert_zone_prediction(self, zone_data: Dict[str, Any]) -> bool:
        """Insert zone prediction into database"""
        try:
            response = self.client.table("zone_predictions").insert([zone_data]).execute()
            self.logger.debug(f"Inserted zone prediction for {zone_data['symbol']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to insert zone prediction: {e}")
            return False
    
    def insert_session_result(self, session_data: Dict[str, Any]) -> bool:
        """Insert trading session result"""
        try:
            response = self.client.table("trading_sessions").insert([session_data]).execute()
            self.logger.debug(f"Inserted session result: {session_data['session_id']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to insert session result: {e}")
            return False
    
    def insert_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """Insert prediction feedback for model improvement"""
        try:
            response = self.client.table("prediction_feedback").insert([feedback_data]).execute()
            self.logger.debug(f"Inserted feedback for prediction {feedback_data.get('prediction_id')}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to insert feedback: {e}")
            return False
    
    def get_recent_predictions(self, symbol: str = "BTC/USDT", limit: int = 20) -> Optional[pd.DataFrame]:
        """Get recent zone predictions with outcomes"""
        try:
            response = (self.client.table("zone_predictions")
                       .select("*")
                       .eq("symbol", symbol)
                       .order("timestamp", desc=True)
                       .limit(limit)
                       .execute())
            
            if not response.data:
                return None
                
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get predictions: {e}")
            return None
    
    def get_prediction_streak(self, symbol: str = "BTC/USDT") -> Dict:
        """Get current prediction streak (winning/losing)"""
        try:
            response = (self.client.table("zone_predictions")
                       .select("zone_hit")
                       .eq("symbol", symbol)
                       .order("timestamp", desc=True)
                       .limit(50)
                       .execute())
            
            if not response.data:
                return {"current_streak": 0, "type": "none", "max_streak": 0}
            
            # Calculate streak
            current_streak = 0
            streak_type = None
            max_streak = 0
            
            for prediction in response.data:
                if prediction['zone_hit'] is None:
                    continue
                    
                if streak_type is None:
                    streak_type = "winning" if prediction['zone_hit'] else "losing"
                    current_streak = 1
                elif (prediction['zone_hit'] and streak_type == "winning") or \
                     (not prediction['zone_hit'] and streak_type == "losing"):
                    current_streak += 1
                else:
                    break
            
            max_streak = max(current_streak, max_streak)
            
            return {
                "current_streak": current_streak,
                "type": streak_type or "none",
                "max_streak": max_streak
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get streak: {e}")
            return {"current_streak": 0, "type": "error", "max_streak": 0}
    
    def get_zone_accuracy_stats(self, symbol: str = "BTC/USDT", days: int = 7) -> Dict:
        """Get zone prediction accuracy statistics"""
        try:
            response = (self.client.table("zone_predictions")
                       .select("*")
                       .eq("symbol", symbol)
                       .execute())
            
            if not response.data:
                return {"accuracy": 0.0, "total": 0, "correct": 0}
            
            total = len(response.data)
            correct = sum(1 for r in response.data if r.get('zone_hit', False))
            accuracy = correct / total if total > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "target_accuracy": 0.80
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get zone accuracy: {e}")
            return {"accuracy": 0.0, "total": 0, "correct": 0}
    
    def cleanup_old_data(self, days_to_keep: int = 7) -> bool:
        """Clean up data older than specified days (free tier optimization)"""
        try:
            from datetime import timedelta
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
            
            # Delete old candles
            self.client.table("candles").delete().lt("created_at", cutoff_date).execute()
            
            # Delete old zone predictions  
            self.client.table("zone_predictions").delete().lt("created_at", cutoff_date).execute()
            
            self.logger.info(f"Cleaned up data older than {days_to_keep} days")
            return True
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return False
    
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
    
    def create_zone_tables_sql(self) -> str:
        """SQL to create zone-related tables for 80% accuracy tracking"""
        return """
        -- Zone predictions table
        CREATE TABLE IF NOT EXISTS zone_predictions (
            id SERIAL PRIMARY KEY,
            symbol TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW(),
            support_zone_min DECIMAL,
            support_zone_max DECIMAL,
            resistance_zone_min DECIMAL,
            resistance_zone_max DECIMAL,
            confidence DECIMAL,
            zone_hit BOOLEAN DEFAULT FALSE,
            actual_price DECIMAL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Trading sessions table
        CREATE TABLE IF NOT EXISTS trading_sessions (
            id SERIAL PRIMARY KEY,
            session_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            predictions_made INTEGER,
            correct_predictions INTEGER,
            accuracy DECIMAL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Prediction feedback table
        CREATE TABLE IF NOT EXISTS prediction_feedback (
            id SERIAL PRIMARY KEY,
            prediction_id TEXT,
            symbol TEXT NOT NULL,
            actual_result TEXT,
            timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_zones_symbol_timestamp 
        ON zone_predictions(symbol, timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_sessions_symbol 
        ON trading_sessions(symbol);
        
        -- Auto-cleanup old data (free tier optimization)
        -- Run this periodically to stay under 500MB limit
        DELETE FROM candles WHERE created_at < NOW() - INTERVAL '7 days';
        DELETE FROM zone_predictions WHERE created_at < NOW() - INTERVAL '30 days';
        """
