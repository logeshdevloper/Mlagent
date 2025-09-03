import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import Config
from src.api.trading_api import SupabaseClient
from src.utils.logger import setup_logger

class SequenceLabeler:
    """Generate labeled sequences for machine learning"""
    
    def __init__(self):
        self.logger = setup_logger(__name__, "labeler.log")
        self.db_client = SupabaseClient()
        
    def create_sequences(self, symbol: str = Config.DEFAULT_SYMBOL) -> Optional[pd.DataFrame]:
        """Create labeled sequences from candle data"""
        
        # Get candle data
        df = self.db_client.get_candles(symbol, limit=5000)
        if df is None or len(df) < Config.SEQUENCE_LENGTH + Config.PREDICTION_HORIZON:
            self.logger.error(
                f"Insufficient data: need at least {Config.SEQUENCE_LENGTH + Config.PREDICTION_HORIZON} "
                f"candles, got {len(df) if df is not None else 0}"
            )
            return None
        
        self.logger.info(f"Creating sequences from {len(df)} candles")
        
        # Prepare feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        sequences = []
        labels = []
        timestamps = []
        
        # Generate sequences
        for i in range(len(df) - Config.SEQUENCE_LENGTH - Config.PREDICTION_HORIZON + 1):
            # Extract sequence
            sequence_data = df.iloc[i:i + Config.SEQUENCE_LENGTH]
            future_candle = df.iloc[i + Config.SEQUENCE_LENGTH]
            
            # Create features (flatten the sequence)
            features = sequence_data[feature_cols].values.flatten()
            
            # Create label (UP=1, DOWN=0)
            current_close = sequence_data.iloc[-1]['close']
            future_close = future_candle['close']
            label = 1 if future_close > current_close else 0
            
            sequences.append(features)
            labels.append(label)
            timestamps.append(sequence_data.iloc[-1]['timestamp'])
        
        # Create DataFrame
        feature_names = []
        for i in range(Config.SEQUENCE_LENGTH):
            for col in feature_cols:
                feature_names.append(f"{col}_{i}")
        
        result_df = pd.DataFrame(sequences, columns=feature_names)
        result_df['label'] = labels
        result_df['sequence_end_timestamp'] = timestamps
        
        self.logger.info(f"Created {len(result_df)} labeled sequences")
        
        # Calculate label distribution
        label_dist = result_df['label'].value_counts()
        self.logger.info(f"Label distribution - UP: {label_dist.get(1, 0)}, DOWN: {label_dist.get(0, 0)}")
        
        return result_df
    
    def save_sequences(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save sequences to CSV"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"labeled_sequences_{timestamp}.csv"
        
        # Create data directory
        Path("data").mkdir(exist_ok=True)
        filepath = Path("data") / filename
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(df)} sequences to {filepath}")
        
        return str(filepath)
