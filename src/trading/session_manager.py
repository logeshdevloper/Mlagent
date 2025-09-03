"""
Session Manager for 1-Hour Manual Trading Sessions
Manages zone predictions and tracks performance for 80% accuracy
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.models.zone_predictor import ZonePredictor
from src.data.binance_fetcher import BinanceFetcher
from src.utils.logger import setup_logger
from config import Config

logger = setup_logger("session_manager")


class TradingSession:
    """Manages a single 1-hour trading session"""
    
    def __init__(self, symbol: str = "BTC/USDT"):
        self.symbol = symbol
        self.config = Config()
        self.zone_predictor = ZonePredictor()
        self.fetcher = BinanceFetcher()
        
        # Session state
        self.session_id = None
        self.start_time = None
        self.end_time = None
        self.active = False
        
        # Zone tracking
        self.current_zones = None
        self.zone_hits = []
        self.signals_given = []
        self.signals_remaining = self.config.MAX_SIGNALS_PER_SESSION
        
        # Performance tracking
        self.predictions_made = 0
        self.correct_predictions = 0
        self.session_accuracy = 0.0
        
    def start_session(self) -> Dict:
        """
        Start a new 1-hour trading session
        Returns initial zones and signals
        """
        try:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.start_time = datetime.now()
            self.end_time = self.start_time + timedelta(hours=1)
            self.active = True
            
            logger.info(f"Starting trading session {self.session_id} for {self.symbol}")
            
            # Get initial zone predictions
            self.current_zones = self.zone_predictor.predict_session_zones(self.symbol)
            
            if self.current_zones and self.current_zones.get('confidence', 0) >= self.config.MIN_CONFIDENCE_THRESHOLD:
                response = {
                    'status': 'success',
                    'session_id': self.session_id,
                    'start_time': self.start_time.isoformat(),
                    'end_time': self.end_time.isoformat(),
                    'symbol': self.symbol,
                    'zones': self.current_zones,
                    'initial_signals': self.current_zones.get('signals', []),
                    'message': f'Session started. Confidence: {self.current_zones["confidence"]:.1%}'
                }
                
                # Track initial signals
                self.signals_given = self.current_zones.get('signals', [])
                self.signals_remaining = self.config.MAX_SIGNALS_PER_SESSION - len(self.signals_given)
                
            else:
                response = {
                    'status': 'warning',
                    'session_id': self.session_id,
                    'message': 'Session started but confidence too low for trading',
                    'zones': self.current_zones,
                    'recommendation': 'Wait for better market conditions'
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            return {
                'status': 'error',
                'message': f'Failed to start session: {str(e)}'
            }
    
    def update_session(self) -> Dict:
        """
        Update session with latest data (called every 15 minutes)
        Checks if zones are still valid and provides new signals if needed
        """
        if not self.active:
            return {'status': 'error', 'message': 'No active session'}
        
        try:
            # Check if session expired
            if datetime.now() > self.end_time:
                return self.end_session()
            
            # Fetch latest price
            session_data = self.fetcher.fetch_session_data(self.symbol)
            if not session_data:
                return {'status': 'error', 'message': 'Failed to fetch latest data'}
            
            current_price = session_data['current_price']
            
            # Check if any zones were hit
            zone_hit = self._check_zone_hits(current_price)
            
            # Recalculate zones if needed (market conditions changed)
            should_recalculate = self._should_recalculate_zones()
            
            if should_recalculate:
                self.current_zones = self.zone_predictor.predict_session_zones(self.symbol)
                new_signals = self.current_zones.get('signals', [])
                
                # Add new signals if we have room
                if self.signals_remaining > 0 and new_signals:
                    new_signals = new_signals[:self.signals_remaining]
                    self.signals_given.extend(new_signals)
                    self.signals_remaining -= len(new_signals)
            else:
                new_signals = []
            
            # Calculate session performance
            self._update_accuracy()
            
            return {
                'status': 'success',
                'session_id': self.session_id,
                'time_remaining': str(self.end_time - datetime.now()),
                'current_price': current_price,
                'zones': self.current_zones,
                'zone_hit': zone_hit,
                'new_signals': new_signals,
                'signals_remaining': self.signals_remaining,
                'session_accuracy': f'{self.session_accuracy:.1%}',
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions
            }
            
        except Exception as e:
            logger.error(f"Session update failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_zone_hits(self, current_price: float) -> Optional[Dict]:
        """Check if current price hit any predicted zones"""
        if not self.current_zones:
            return None
        
        zone_hit = None
        
        # Check support zone
        support_zone = self.current_zones['support_zone']
        if support_zone['range'][0] <= current_price <= support_zone['range'][1]:
            zone_hit = {
                'type': 'SUPPORT_HIT',
                'zone': support_zone,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            self.zone_hits.append(zone_hit)
            self.correct_predictions += 1
            logger.info(f"Support zone hit at {current_price}")
        
        # Check resistance zone
        resistance_zone = self.current_zones['resistance_zone']
        if resistance_zone['range'][0] <= current_price <= resistance_zone['range'][1]:
            zone_hit = {
                'type': 'RESISTANCE_HIT',
                'zone': resistance_zone,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            self.zone_hits.append(zone_hit)
            self.correct_predictions += 1
            logger.info(f"Resistance zone hit at {current_price}")
        
        if zone_hit:
            self.predictions_made += 1
        
        return zone_hit
    
    def _should_recalculate_zones(self) -> bool:
        """Determine if zones need recalculation"""
        # Recalculate based on configured interval or if confidence dropped
        if not self.current_zones:
            return True
        
        zone_age = datetime.now() - datetime.fromisoformat(self.current_zones['timestamp'])
        recalc_seconds = self.config.ZONE_RECALCULATION_MINUTES * 60
        if zone_age.total_seconds() > recalc_seconds:
            return True
        
        # Recalculate if market conditions changed significantly
        # This would check for volatility spikes, news events, etc.
        return False
    
    def _update_accuracy(self):
        """Update session accuracy metrics"""
        if self.predictions_made > 0:
            self.session_accuracy = self.correct_predictions / self.predictions_made
    
    def end_session(self) -> Dict:
        """End the current trading session and provide summary"""
        if not self.active:
            return {'status': 'error', 'message': 'No active session to end'}
        
        self.active = False
        session_duration = datetime.now() - self.start_time
        
        # Calculate final accuracy
        self._update_accuracy()
        
        # Determine if session met 80% accuracy target
        target_met = self.session_accuracy >= self.config.MIN_CONFIDENCE_THRESHOLD
        
        summary = {
            'status': 'completed',
            'session_id': self.session_id,
            'duration': str(session_duration),
            'symbol': self.symbol,
            'performance': {
                'predictions_made': self.predictions_made,
                'correct_predictions': self.correct_predictions,
                'accuracy': f'{self.session_accuracy:.1%}',
                'target_accuracy': f'{self.config.MIN_CONFIDENCE_THRESHOLD:.0%}',
                'target_met': target_met
            },
            'zones_hit': len(self.zone_hits),
            'signals_given': len(self.signals_given),
            'zone_hits_detail': self.zone_hits,
            'recommendation': self._get_session_recommendation()
        }
        
        logger.info(f"Session {self.session_id} ended. Accuracy: {self.session_accuracy:.1%}")
        
        # Save session data for analysis
        self._save_session_data(summary)
        
        return summary
    
    def _get_session_recommendation(self) -> str:
        """Provide recommendation based on session performance"""
        if self.session_accuracy >= 0.8:
            return "Excellent session! Zone predictions were highly accurate."
        elif self.session_accuracy >= 0.7:
            return "Good session. Consider waiting for higher confidence zones."
        elif self.session_accuracy >= 0.6:
            return "Average session. Review zone strength indicators."
        else:
            return "Below target. Consider adjusting strategy or waiting for better conditions."
    
    def _save_session_data(self, summary: Dict):
        """Save session data for future analysis"""
        try:
            filename = f"data/sessions/session_{self.session_id}.json"
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Session data saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save session data: {e}")
    
    def get_session_status(self) -> Dict:
        """Get current session status"""
        if not self.active:
            return {'status': 'inactive', 'message': 'No active session'}
        
        time_remaining = self.end_time - datetime.now()
        
        return {
            'status': 'active',
            'session_id': self.session_id,
            'symbol': self.symbol,
            'time_remaining': str(time_remaining),
            'time_elapsed': str(datetime.now() - self.start_time),
            'current_zones': self.current_zones,
            'signals_remaining': self.signals_remaining,
            'current_accuracy': f'{self.session_accuracy:.1%}',
            'zone_hits': len(self.zone_hits)
        }


class SessionManager:
    """Manages multiple trading sessions"""
    
    def __init__(self):
        self.active_sessions = {}
        self.session_history = []
        self.logger = setup_logger("session_manager")
    
    def create_session(self, symbol: str = "BTC/USDT") -> Dict:
        """Create a new trading session"""
        if symbol in self.active_sessions:
            return {
                'status': 'error',
                'message': f'Active session already exists for {symbol}'
            }
        
        session = TradingSession(symbol)
        result = session.start_session()
        
        if result['status'] == 'success':
            self.active_sessions[symbol] = session
        
        return result
    
    def update_session(self, symbol: str = "BTC/USDT") -> Dict:
        """Update an active session"""
        if symbol not in self.active_sessions:
            return {
                'status': 'error',
                'message': f'No active session for {symbol}'
            }
        
        return self.active_sessions[symbol].update_session()
    
    def end_session(self, symbol: str = "BTC/USDT") -> Dict:
        """End an active session"""
        if symbol not in self.active_sessions:
            return {
                'status': 'error',
                'message': f'No active session for {symbol}'
            }
        
        result = self.active_sessions[symbol].end_session()
        
        # Move to history
        self.session_history.append(result)
        del self.active_sessions[symbol]
        
        return result
    
    def get_all_sessions(self) -> Dict:
        """Get status of all sessions"""
        return {
            'active': {
                symbol: session.get_session_status()
                for symbol, session in self.active_sessions.items()
            },
            'history_count': len(self.session_history),
            'recent_history': self.session_history[-5:] if self.session_history else []
        }