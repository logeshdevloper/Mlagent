import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
from src.models.sequence_labeler import SequenceLabeler
from src.data.binance_fetcher import BinanceFetcher
from src.utils.logger import setup_logger
import asyncio
import aiohttp
import time

class FastModelPredictor:
    """Optimized predictor for real-time 1-minute predictions"""
    
    def __init__(self):
        self.model = joblib.load("artifacts/model_v1_lgb.pkl")
        self.labeler = SequenceLabeler()
        self.fetcher = BinanceFetcher()
        self.logger = setup_logger("fast_predictor")
        
        # Cache for performance
        self.candle_cache = {}
        self.last_prediction_time = {}
        self.prediction_history = []
        
    def predict_next_candle(self, symbol: str = "BTC/USDT") -> Dict:
        """Fast prediction optimized for real-time use"""
        start_time = time.time()
        
        try:
            # 1. Fast data fetching (cached + optimized)
            df = self._fetch_candles_fast(symbol)
            if df is None or len(df) < 180:  # Back to original 180 candles for better accuracy
                return {"error": f"Insufficient candle data: got {len(df) if df is not None else 0}, need at least 180", "status": "failed"}
            
            # 2. Optimized feature extraction
            features = self._extract_features_fast(df.tail(180))  # Back to original 180 candles for better accuracy
            if features is None:
                return {"error": "Feature extraction failed", "status": "failed"}
            
            # 3. Fast prediction
            prediction_proba = self.model.predict_proba([features])[0]
            prediction = int(prediction_proba[1] > 0.5)
            confidence = max(prediction_proba)
            
            # 4. Calculate predicted price and timestamps
            current_price = df.iloc[-1]['close']
            current_timestamp = df.iloc[-1]['timestamp']
            
            # Calculate predicted price based on direction and confidence
            predicted_price = self._calculate_predicted_price(current_price, prediction, confidence)
            
            # Calculate actual timestamp (next minute)
            actual_timestamp = current_timestamp + pd.Timedelta(minutes=1)
            
            # 5. Build response
            prediction_result = {
                "prediction": "UP" if prediction == 1 else "DOWN",
                "confidence": round(float(confidence), 4),
                "probability_up": round(float(prediction_proba[1]), 4),
                "probability_down": round(float(prediction_proba[0]), 4),
                "current_price": float(current_price),
                "predicted_price": float(predicted_price),
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction_timestamp": current_timestamp.isoformat(),
                "actual_timestamp": actual_timestamp.isoformat(),
                "model_version": "v1_lgb_fast",
                "confidence_level": self._get_confidence_level(confidence),
                "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                "cache_hit": symbol in self.candle_cache
            }
            
            # 6. Store prediction in database
            self._store_prediction(prediction_result)
            
            # 7. Update cache
            self._update_cache(symbol, df)
            self._log_prediction(prediction_result)
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Fast prediction failed for {symbol}: {e}")
            return {"error": str(e), "status": "failed", "processing_time_ms": round((time.time() - start_time) * 1000, 2)}
    
    def _fetch_candles_fast(self, symbol: str) -> Optional[pd.DataFrame]:
        """Optimized candle fetching with caching"""
        try:
            # Check cache first
            if symbol in self.candle_cache:
                cached_data = self.candle_cache[symbol]
                # Use cache if data is less than 30 seconds old
                if time.time() - cached_data['timestamp'] < 30:
                    return cached_data['data']
            
            # Fetch from database (much faster than API calls)
            result = self.fetcher.db_client.client \
                .table("candles") \
                .select("*") \
                .eq("symbol", symbol) \
                .order("timestamp", desc=True) \
                .limit(200) \
                .execute()
            
            if result.data:
                df = pd.DataFrame(result.data)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Cache the result
                self.candle_cache[symbol] = {
                    'data': df,
                    'timestamp': time.time()
                }
                
                return df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Fast candle fetch failed: {e}")
            return None
    
    def _extract_features_fast(self, candle_window: pd.DataFrame) -> Optional[np.ndarray]:
        """Optimized feature extraction"""
        try:
            # Use pre-computed technical indicators if available
            df_with_indicators = self.labeler.add_technical_indicators(candle_window.copy())
            
            # Extract only essential features for speed
            feature_cols = [col for col in df_with_indicators.columns 
                          if col not in ['timestamp', 'symbol', 'created_at']]
            features = df_with_indicators[feature_cols].values.flatten()
            
            # Ensure we have exactly 5940 features (what the model expects)
            expected_features = 5940
            if len(features) != expected_features:
                self.logger.warning(f"Feature count mismatch: got {len(features)}, expected {expected_features}")
                # Pad or truncate to match expected size
                if len(features) > expected_features:
                    features = features[:expected_features]
                else:
                    # Pad with zeros if we have fewer features
                    padding = np.zeros(expected_features - len(features))
                    features = np.concatenate([features, padding])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Fast feature extraction failed: {e}")
            return None
    
    def _update_cache(self, symbol: str, df: pd.DataFrame):
        """Update cache with new data"""
        self.candle_cache[symbol] = {
            'data': df,
            'timestamp': time.time()
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Categorize prediction confidence"""
        if confidence >= 0.7:
            return "HIGH"
        elif confidence >= 0.6:
            return "MEDIUM"
        elif confidence >= 0.55:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _calculate_predicted_price(self, current_price: float, prediction: int, confidence: float) -> float:
        """Calculate predicted price based on direction and confidence"""
        # Base movement percentage (0.1% to 0.5% based on confidence)
        base_movement = 0.001 + (confidence - 0.5) * 0.008  # 0.1% to 0.5%
        
        if prediction == 1:  # UP
            predicted_price = current_price * (1 + base_movement)
        else:  # DOWN
            predicted_price = current_price * (1 - base_movement)
            
        return round(predicted_price, 2)
    
    def _store_prediction(self, prediction: Dict):
        """Store prediction in database"""
        try:
            prediction_data = {
                "symbol": prediction["symbol"],
                "prediction_timestamp": prediction["prediction_timestamp"],
                "actual_timestamp": prediction["actual_timestamp"],
                "current_price": prediction["current_price"],
                "predicted_direction": prediction["prediction"],
                "predicted_price": prediction["predicted_price"],
                "confidence": prediction["confidence"],
                "probability_up": prediction["probability_up"],
                "probability_down": prediction["probability_down"],
                "model_version": prediction["model_version"],
                "processing_time_ms": prediction["processing_time_ms"],
                "cache_hit": prediction["cache_hit"]
            }
            
            self.fetcher.db_client.insert_prediction(prediction_data)
            self.logger.info(f"Stored prediction for {prediction['symbol']}: {prediction['prediction']} at ${prediction['predicted_price']}")
            
        except Exception as e:
            self.logger.error(f"Failed to store prediction: {e}")
    
    def _log_prediction(self, prediction: Dict):
        """Log prediction for tracking"""
        self.prediction_history.append({
            "timestamp": prediction["timestamp"],
            "prediction": prediction["prediction"],
            "confidence": prediction["confidence"],
            "symbol": prediction["symbol"],
            "predicted_price": prediction.get("predicted_price", 0),
            "processing_time_ms": prediction.get("processing_time_ms", 0)
        })
        
        # Keep only last 100 predictions
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.prediction_history:
            return {"avg_processing_time_ms": 0, "total_predictions": 0}
        
        times = [p.get("processing_time_ms", 0) for p in self.prediction_history]
        return {
            "avg_processing_time_ms": float(round(np.mean(times), 2)),
            "max_processing_time_ms": float(round(np.max(times), 2)),
            "min_processing_time_ms": float(round(np.min(times), 2)),
            "total_predictions": int(len(self.prediction_history)),
            "cache_hit_rate": float(len([p for p in self.prediction_history if p.get("cache_hit", False)]) / len(self.prediction_history))
        }
    
    def get_stored_predictions(self, symbol: str = "BTC/USDT", limit: int = 20) -> List[Dict]:
        """Get recent predictions from database"""
        try:
            df = self.fetcher.db_client.get_recent_predictions(symbol, limit)
            if df is None:
                return []
            
            predictions = []
            for _, row in df.iterrows():
                predictions.append({
                    "id": row["id"],
                    "symbol": row["symbol"],
                    "prediction_timestamp": row["prediction_timestamp"].isoformat(),
                    "actual_timestamp": row["actual_timestamp"].isoformat(),
                    "current_price": float(row["current_price"]),
                    "predicted_price": float(row["predicted_price"]) if row["predicted_price"] else None,
                    "predicted_direction": row["predicted_direction"],
                    "confidence": float(row["confidence"]),
                    "actual_price": float(row["actual_price"]) if row["actual_price"] else None,
                    "actual_direction": row["actual_direction"] if row["actual_direction"] else None,
                    "prediction_correct": row["prediction_correct"] if row["prediction_correct"] is not None else None,
                    "processing_time_ms": float(row["processing_time_ms"])
                })
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Failed to get stored predictions: {e}")
            return []
    
    def get_model_metrics(self) -> Dict:
        """Return model metrics with performance stats"""
        perf_stats = self.get_performance_stats()
        
        return {
            "model_info": {
                "version": "v1_lgb_fast",
                "accuracy": float(0.529),
                "training_date": "2025-08-04",
                "features_count": int(5940),
                "avg_processing_time_ms": perf_stats["avg_processing_time_ms"]
            },
            "performance": perf_stats,
            "live_performance": {
                "total_predictions": int(len(self.prediction_history)),
                "recent_predictions": self.prediction_history[-10:] if self.prediction_history else [],
                "cache_status": {symbol: "HIT" if symbol in self.candle_cache else "MISS" 
                               for symbol in ["BTC/USDT", "ETH/USDT", "ADA/USDT"]}
            },
            "model_capabilities": {
                "down_detection_recall": float(0.74),
                "up_detection_recall": float(0.32),
                "best_at": "Detecting market drops",
                "confidence_threshold": "Use predictions with confidence > 0.6",
                "real_time_ready": bool(perf_stats["avg_processing_time_ms"] < 1000)  # Under 1 second
            }
        }
