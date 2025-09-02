# src/models/sequence_labeler.py

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Optional, Tuple
from src.api.supabase_client import SupabaseClient
from src.utils.logger import setup_logger
from config import Config

logger = setup_logger("sequence_labeler")


class SequenceLabeler:
    """Enhanced feature engineering with technical indicators for ML training data"""
    
    def __init__(self):
        self.db = SupabaseClient()
        self.config = Config()
    
    def fetch_candle_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data for a symbol
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            DataFrame with candle data or None if failed
        """
        logger.info(f"📦 Fetching candle data for: {symbol}")
        
        try:
            result = self.db.client \
                .table("candles") \
                .select("timestamp,open,high,low,close,volume") \
                .eq("symbol", symbol) \
                .order("timestamp", desc=False) \
                .limit(self.config.MAX_ROWS_FOR_LABELING) \
                .execute()
            
            if not result.data:
                logger.warning(f"❌ No data found for symbol: {symbol}")
                return None
            
            df = pd.DataFrame(result.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert price columns to float
            price_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data for {symbol}: {e}")
            return None
    
    def calculate_sma(self, data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window, min_periods=1).mean()
    
    def calculate_ema(self, data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    
    def calculate_macd(self, data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = self.calculate_ema(data, fast)
        ema_slow = self.calculate_ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, data: pd.Series, window: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = self.calculate_sma(data, window)
        std = data.rolling(window=window, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=window, min_periods=1).min()
        highest_high = high.rolling(window=window, min_periods=1).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent = k_percent.fillna(50)
        d_percent = k_percent.rolling(window=3, min_periods=1).mean()
        return k_percent, d_percent
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        return atr
    
    def calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window, min_periods=1).max()
        lowest_low = low.rolling(window=window, min_periods=1).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r.fillna(-50)
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators to the DataFrame"""
        logger.info("🔧 Calculating technical indicators...")
        
        # Price-based indicators
        df['sma_5'] = self.calculate_sma(df['close'], 5)
        df['sma_10'] = self.calculate_sma(df['close'], 10)
        df['sma_20'] = self.calculate_sma(df['close'], 20)
        df['sma_50'] = self.calculate_sma(df['close'], 50)
        
        df['ema_5'] = self.calculate_ema(df['close'], 5)
        df['ema_10'] = self.calculate_ema(df['close'], 10)
        df['ema_20'] = self.calculate_ema(df['close'], 20)
        
        # RSI
        df['rsi'] = self.calculate_rsi(df['close'])
        
        # MACD
        macd, macd_signal, macd_hist = self.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = bb_upper - bb_lower
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        
        # Stochastic
        stoch_k, stoch_d = self.calculate_stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        
        # ATR
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = self.calculate_williams_r(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = self.calculate_sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price ratios and differences
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Price momentum
        df['price_change'] = df['close'].pct_change()
        df['price_change_5'] = df['close'].pct_change(5)
        
        # Support/Resistance levels
        df['resistance_20'] = df['high'].rolling(window=20, min_periods=1).max()
        df['support_20'] = df['low'].rolling(window=20, min_periods=1).min()
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"✅ Added {len([col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])} technical indicators")
        return df
    
    def _create_features(self, window: pd.DataFrame) -> np.ndarray:
        """
        Create flattened feature vector from candle window with technical indicators
        
        Args:
            window: DataFrame slice of candles with indicators
            
        Returns:
            Flattened numpy array of features
        """
        # Select all feature columns (exclude timestamp and metadata)
        feature_cols = [col for col in window.columns if col not in ['timestamp', 'symbol', 'created_at']]
        features = window[feature_cols].values
        return features.flatten()
    
    def _create_label(self, current_close: float, next_close: float) -> int:
        """
        Create binary label based on price direction
        
        Args:
            current_close: Current candle close price
            next_close: Next candle close price
            
        Returns:
            1 if price goes up, 0 if down
        """
        return 1 if next_close > current_close else 0
    
    def create_sequences_and_labels(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Generate feature sequences with technical indicators and labels for ML training
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            DataFrame with flattened sequences and labels
        """
        # Fetch raw data
        df = self.fetch_candle_data(symbol)
        if df is None:
            return None
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Check if we have enough data
        min_required = self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON
        if len(df) < min_required:
            logger.warning(f"⚠️ Insufficient data: {len(df)} < {min_required} required")
            return None
        
        logger.info(f"🔧 Creating sequences with {self.config.SEQUENCE_LENGTH} lookback...")
        
        sequences = []
        labels = []
        
        # Create sliding windows
        for i in range(len(df) - self.config.SEQUENCE_LENGTH - self.config.PREDICTION_HORIZON + 1):
            # Extract sequence window
            window = df.iloc[i:i + self.config.SEQUENCE_LENGTH]
            
            # Get current and next close prices for labeling
            current_close = window.iloc[-1]['close']
            next_close = df.iloc[i + self.config.SEQUENCE_LENGTH]['close']
            
            # Create features and label
            features = self._create_features(window)
            label = self._create_label(current_close, next_close)
            
            sequences.append(features)
            labels.append(label)
        
        # Create final DataFrame
        feature_df = pd.DataFrame(sequences)
        feature_df['label'] = labels
        
        # Add metadata
        feature_df['symbol'] = symbol
        feature_df['created_at'] = datetime.now().isoformat()
        
        logger.info(f"✅ Created {len(feature_df)} sequences for {symbol}")
        logger.info(f"📊 Label distribution - UP: {sum(labels)}, DOWN: {len(labels) - sum(labels)}")
        logger.info(f"🎯 Feature count per sequence: {len(sequences[0])}")
        
        return feature_df
    
    def save_sequences(self, df: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """
        Save sequences to CSV file
        
        Args:
            df: DataFrame with sequences and labels
            output_file: Custom output filename (optional)
            
        Returns:
            Path to saved file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = df['symbol'].iloc[0].replace('/', '_')
            output_file = f"{self.config.OUTPUT_DIR}/sequences_{symbol}_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        logger.info(f"💾 Saved {len(df)} sequences to: {output_file}")
        return output_file
    
    def process_symbol(self, symbol: str, output_file: Optional[str] = None) -> Optional[str]:
        """
        Complete processing pipeline for a symbol with technical indicators
        
        Args:
            symbol: Trading pair symbol
            output_file: Custom output filename (optional)
            
        Returns:
            Path to saved file or None if failed
        """
        logger.info(f"🚀 Starting enhanced feature engineering for {symbol}")
        
        # Create sequences and labels with technical indicators
        sequences_df = self.create_sequences_and_labels(symbol)
        
        if sequences_df is None:
            logger.error(f"❌ Failed to create sequences for {symbol}")
            return None
        
        # Save to file
        filepath = self.save_sequences(sequences_df, output_file)
        
        logger.info(f"🎯 Successfully processed {symbol} with technical indicators → {filepath}")
        return filepath