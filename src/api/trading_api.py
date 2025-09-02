from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from src.models.predictor import AdvancedModelPredictor
from src.utils.logger import setup_logger
import traceback
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# Initialize predictor and logger
predictor = AdvancedModelPredictor()
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
        "service": "Quotex-AI Trading API",
        "version": "1.0.0",
        "model_loaded": True
    })

@app.route('/predict', methods=['POST', 'GET'])
def predict_next_candle():
    """Advanced prediction endpoint with comprehensive response"""
    try:
        # Get symbol from request
        if request.method == 'POST':
            data = request.get_json() or {}
            symbol = data.get('symbol', 'BTC/USDT')
        else:
            symbol = request.args.get('symbol', 'BTC/USDT')
        
        logger.info(f"Prediction request for {symbol}")
        
        # Make prediction
        result = predictor.predict_next_candle(symbol)
        
        if "error" in result:
            logger.error(f"Prediction failed: {result['error']}")
            return jsonify(result), 400
        
        logger.info(f"Prediction successful: {result['prediction']} with {result['confidence']:.3f} confidence")
        
        return jsonify({
            "status": "success",
            "data": result,
            "usage_tips": {
                "high_confidence": "Use predictions with confidence > 0.6",
                "model_strength": "Better at detecting DOWN moves (74% recall)",
                "recommended_use": "Combine with your own analysis"
            }
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
            "timestamp": data.get('timestamp'),
            "symbol": data.get('symbol')
        }
        
        # TODO: Store in database for retraining
        logger.info(f"Feedback received: {feedback_entry}")
        
        return jsonify({
            "status": "success",
            "message": "Feedback recorded for model improvement"
        })
        
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
