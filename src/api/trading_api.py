from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from src.models.predictor import AdvancedModelPredictor
from src.models.zone_predictor import ZonePredictor
from src.trading.session_manager import SessionManager
from src.data.binance_fetcher import BinanceFetcher
from src.api.supabase_client import SupabaseClient
from src.utils.logger import setup_logger
import traceback
import os
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize components
predictor = AdvancedModelPredictor()
zone_predictor = ZonePredictor()
session_manager = SessionManager()
fetcher = BinanceFetcher()
db_client = SupabaseClient()
logger = setup_logger("trading_api")

@app.route('/', methods=['GET'])
def index():
    """Serve the dashboard HTML"""
    return send_from_directory('static', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        "status": "running",
        "service": "Zone-Based Trading API (80% Accuracy Target)",
        "version": "2.0.0",
        "model_loaded": True,
        "mode": "zone_prediction",
        "target_accuracy": "80%"
    })

@app.route('/predict', methods=['POST', 'GET'])
def predict_next_candle():
    """Zone-based prediction endpoint (80% accuracy target)"""
    try:
        # Get symbol from request
        if request.method == 'POST':
            data = request.get_json() or {}
            symbol = data.get('symbol', 'BTC/USDT')
        else:
            symbol = request.args.get('symbol', 'BTC/USDT')
        
        logger.info(f"Zone prediction request for {symbol}")
        
        # Use zone-based prediction for 80% accuracy
        zones = zone_predictor.predict_session_zones(symbol)
        
        # Format response for compatibility
        if zones.get('confidence', 0) >= 0.80:
            # High confidence zone prediction
            recommendation = zones.get('recommendation', {})
            if recommendation.get('action') == 'BUY':
                prediction = 'UP'
            elif recommendation.get('action') == 'SELL':
                prediction = 'DOWN'
            else:
                prediction = 'WAIT'
            
            result = {
                "prediction": prediction,
                "confidence": zones['confidence'],
                "support_zone": zones['support_zone']['range'],
                "resistance_zone": zones['resistance_zone']['range'],
                "current_price": zones['current_price'],
                "recommendation": recommendation,
                "zones": zones
            }
            
            logger.info(f"Zone prediction: {prediction} with {zones['confidence']:.1%} confidence")
            
            return jsonify({
                "status": "success",
                "data": result,
                "usage_tips": {
                    "high_confidence": "Trade only when confidence > 80%",
                    "zone_based": "Using support/resistance zones for higher accuracy",
                    "recommended_use": "Enter at zone boundaries, exit at opposite zone"
                }
            })
        else:
            # Low confidence - don't trade
            return jsonify({
                "status": "low_confidence",
                "data": {
                    "prediction": "WAIT",
                    "confidence": zones.get('confidence', 0),
                    "reason": "Confidence below 80% threshold",
                    "zones": zones
                },
                "message": "Market conditions not favorable for high-accuracy prediction"
            })
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "Internal server error"
        }), 500

@app.route('/metrics', methods=['GET'])
def get_model_metrics():
    """Comprehensive model performance metrics"""
    try:
        metrics = predictor.get_model_metrics()
        return jsonify({
            "status": "success",
            "data": metrics
        })
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit prediction feedback for model improvement"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        actual_result = data.get('actual_result')  # 'UP' or 'DOWN'
        
        # Store feedback for future model retraining
        feedback_entry = {
            "prediction_id": prediction_id,
            "actual_result": actual_result,
            "timestamp": data.get('timestamp', datetime.now().isoformat()),
            "symbol": data.get('symbol', 'BTC/USDT')
        }
        
        # Store in database for retraining
        success = db_client.insert_feedback(feedback_entry)
        
        if success:
            logger.info(f"Feedback stored: {feedback_entry}")
            return jsonify({
                "status": "success",
                "message": "Feedback recorded for model improvement",
                "data": feedback_entry
            })
        else:
            logger.error(f"Failed to store feedback: {feedback_entry}")
            return jsonify({
                "status": "error",
                "message": "Failed to store feedback"
            }), 500
        
    except Exception as e:
        logger.error(f"Feedback endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    """Predict multiple symbols at once"""
    try:
        data = request.get_json()
        symbols = data.get('symbols', ['BTC/USDT'])
        
        results = {}
        for symbol in symbols:
            results[symbol] = predictor.predict_next_candle(symbol)
        
        return jsonify({
            "status": "success",
            "data": results,
            "count": len(results)
        })
        
    except Exception as e:
        logger.error(f"Bulk predict error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/test_bias', methods=['GET'])
def test_model_bias():
    """Test model bias by making multiple predictions"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        bias_result = predictor.test_model_bias(symbol)
        
        if "error" in bias_result:
            return jsonify({
                "status": "error",
                "error": bias_result["error"]
            }), 400
        
        return jsonify({
            "status": "success",
            "data": bias_result,
            "recommendation": "Consider adjusting prediction threshold if bias is too high"
        })
        
    except Exception as e:
        logger.error(f"Test bias error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

# Zone-based prediction endpoints
@app.route('/predict_zones', methods=['GET', 'POST'])
def predict_zones():
    """Predict support/resistance zones for next hour (80% accuracy target)"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT') if request.method == 'GET' else request.get_json().get('symbol', 'BTC/USDT')
        
        logger.info(f"Zone prediction request for {symbol}")
        
        # Get zone predictions
        zones = zone_predictor.predict_session_zones(symbol)
        
        if zones.get('confidence', 0) >= 0.80:
            return jsonify({
                "status": "success",
                "data": zones,
                "tradeable": True,
                "message": f"High confidence zones detected ({zones['confidence']:.1%})"
            })
        else:
            return jsonify({
                "status": "low_confidence",
                "data": zones,
                "tradeable": False,
                "message": "Confidence below 80% threshold. Wait for better conditions."
            })
            
    except Exception as e:
        logger.error(f"Zone prediction error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/session/start', methods=['POST'])
def start_session():
    """Start a new 1-hour trading session"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTC/USDT')
        
        result = session_manager.create_session(symbol)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Session start error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/session/update', methods=['GET'])
def update_session():
    """Update current trading session"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        result = session_manager.update_session(symbol)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Session update error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/session/end', methods=['POST'])
def end_session():
    """End current trading session and get summary"""
    try:
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTC/USDT')
        
        result = session_manager.end_session(symbol)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Session end error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/session/status', methods=['GET'])
def session_status():
    """Get status of all sessions"""
    try:
        result = session_manager.get_all_sessions()
        return jsonify({
            "status": "success",
            "data": result
        })
        
    except Exception as e:
        logger.error(f"Session status error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/latest-candle', methods=['GET'])
def get_latest_candle():
    """Get the latest candle data"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        candle = fetcher.fetch_latest_candle(symbol)
        
        if candle:
            return jsonify({
                "status": "success",
                "data": candle
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to fetch candle"
            }), 500
            
    except Exception as e:
        logger.error(f"Latest candle error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/zone_accuracy', methods=['GET'])
def get_zone_accuracy():
    """Get current zone prediction accuracy stats"""
    try:
        stats = zone_predictor.get_zone_accuracy_stats()
        return jsonify({
            "status": "success",
            "data": stats,
            "message": f"Current accuracy: {stats['current_accuracy']:.1%} (Target: 80%)"
        })
        
    except Exception as e:
        logger.error(f"Zone accuracy error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/streak', methods=['GET'])
def get_prediction_streak():
    """Get current prediction winning/losing streak"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        
        # Get streak from database
        streak_data = db_client.get_prediction_streak(symbol)
        
        # Add recommendation based on streak
        if streak_data['type'] == 'winning' and streak_data['current_streak'] >= 5:
            streak_data['recommendation'] = "Strong performance - maintain strategy"
        elif streak_data['type'] == 'losing' and streak_data['current_streak'] >= 3:
            streak_data['recommendation'] = "Consider pausing - review market conditions"
        else:
            streak_data['recommendation'] = "Normal variance - continue monitoring"
        
        return jsonify({
            "status": "success",
            "data": streak_data,
            "symbol": symbol
        })
        
    except Exception as e:
        logger.error(f"Streak endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    """Get recent zone predictions with outcomes"""
    try:
        symbol = request.args.get('symbol', 'BTC/USDT')
        limit = int(request.args.get('limit', 20))
        
        # Get recent predictions from database
        predictions_df = db_client.get_recent_predictions(symbol, limit)
        
        if predictions_df is not None and not predictions_df.empty:
            # Convert to list of dicts for JSON response
            predictions = predictions_df.to_dict('records')
            
            # Calculate summary stats
            total = len(predictions)
            hits = sum(1 for p in predictions if p.get('zone_hit', False))
            accuracy = (hits / total * 100) if total > 0 else 0
            
            return jsonify({
                "status": "success",
                "data": {
                    "predictions": predictions,
                    "summary": {
                        "total": total,
                        "hits": hits,
                        "accuracy": f"{accuracy:.1f}%",
                        "target": "80%"
                    }
                },
                "symbol": symbol
            })
        else:
            return jsonify({
                "status": "success",
                "data": {
                    "predictions": [],
                    "summary": {
                        "total": 0,
                        "hits": 0,
                        "accuracy": "0%",
                        "target": "80%"
                    }
                },
                "symbol": symbol,
                "message": "No predictions found"
            })
            
    except Exception as e:
        logger.error(f"Predictions endpoint error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
