```markdown
# Quotex-AI 🧠📊

An autonomous AI trading system designed to predict the next 1-minute candle (UP/DOWN) based on real candlestick data — no screenshots, no guesswork.

---

## 🔍 What This Project Does

1. 📈 **Collects real-time 1-minute candle data** from Binance
2. 🧠 **Extracts features** like streaks, wicks, RSI, patterns from 180-candle sequences
3. 🤖 **Predicts the next candle direction** using ML models (LightGBM / XGBoost)
4. 🔁 **Learns from feedback** when predictions are wrong (manual correction or re-analysis)
5. 🎯 **Tracks streaks and accuracy** to adapt and improve continuously

---

## 🛠️ Tech Stack

* **Backend**: Flask
* **ML Models**: LightGBM (primary), XGBoost, LSTM (future)
* **Data Source**: Binance 1m OHLCV API
* **Optimization**: Optuna hyperparameter tuning
* **Database**: Supabase
* **Agent Pipeline**: LangGraph (future)
* **Dashboard**: Plotly / Streamlit (future)

---

## 🚀 Current Status: Phase 4 Completed ✅

**Latest Achievement**: Complete Flask API with Beautiful Web Dashboard
- **Model Accuracy**: 52.9% on test data
- **Model Type**: LightGBM Classifier
- **Dataset**: 820 sequences × 5,943 features
- **Performance**: Better at detecting DOWN moves (74% recall) vs UP moves (32% recall)
- **Dashboard**: Real-time predictions, metrics, and feedback system

Next: **Phase 5 - Advanced Features & Optimization**

---

## ✅ Completed Phases

### Phase 0 - Setup & Planning ✅
- Project folder and modular structure
- Python environment and requirements
- Tech stack finalized

### Phase 1 - Data Pipeline ✅  
- Real-time Binance 1m candle fetcher
- Supabase client for database operations
- Centralized logging utility
- CLI interface for data collection and status

### Phase 2 - Feature Engineering ✅
- Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- 180-candle sequence windowing
- Sequence labeling for ML training
- Feature extraction pipeline (5,943 features per sequence)

### Phase 3 - Model Training & Evaluation ✅
- LightGBM classifier implementation
- Optuna hyperparameter optimization
- Time-series cross-validation
- Model persistence and evaluation metrics
- **Final Model**: 52.9% accuracy, saved to `artifacts/model_v1_lgb.pkl`

---

## 🛠️ In Progress / To Do

### Phase 4 - Flask API Integration ✅
- ✅ `/predict` endpoint for real-time predictions
- ✅ `/feedback` endpoint for model improvement
- ✅ `/metrics` endpoint for performance tracking
- ✅ Live candle integration with existing Binance fetcher
- ✅ **Beautiful Web Dashboard** with real-time predictions and metrics

### Future Phases
- Feedback loop for learning from mistakes
- Dashboard (Plotly/Streamlit)
- Agent pipeline (LangGraph)
- LSTM/Transformer models
- Automated trading integration
- Performance monitoring

---

## 📁 Project Structure

```
src/
├── api/           → Flask endpoints & Supabase client
├── data/          → Candle fetcher & datasets  
├── models/        → Training, evaluation & prediction logic
│   ├── train_model.py      → LightGBM training script
│   ├── tuning_optuna.py    → Hyperparameter optimization
│   ├── eval_model.py       → Model evaluation & metrics
│   └── sequence_labeler.py → Feature engineering pipeline
├── core/          → Agent pipeline (future)
├── utils/         → Config, logging, helpers
└── artifacts/     → Trained models & evaluation results
```

---

## 🖥️ CLI Commands

### Main CLI Interface (`main.py`)

#### Data Collection Commands
```bash
# Start real-time data collection
python main.py collect [--symbol SYMBOL] [--max-iterations MAX_ITERATIONS]

# Store historical data (last 7 days by default)
python main.py store_past_data [--symbol SYMBOL] [--days DAYS]

# Check data status
python main.py status [--symbol SYMBOL]

# Test connections
python main.py test
```

#### Feature Engineering Commands
```bash
# Generate labeled sequences (creates CSV file)
python main.py label [--symbol SYMBOL] [--output OUTPUT]
```

#### API Commands
```bash
# Start Flask API server with dashboard
python main.py api [--host HOST] [--port PORT] [--debug]
```

### Training Commands (Direct Script Execution)

#### Basic Model Training
```bash
# Train LightGBM model
python src/models/train_model.py
```
- Loads data from `data/firstdataset.csv`
- Trains LightGBM classifier
- Saves model to `artifacts/model_v1_lgb.pkl`

#### Hyperparameter Optimization
```bash
# Run Optuna hyperparameter optimization
python src/models/tuning_optuna.py
```
- Runs 50 trials of hyperparameter optimization
- Saves best parameters to `artifacts/best_params_optuna_[timestamp].json`
- Saves study object for later use

#### Model Evaluation
```bash
# Evaluate trained model
python src/models/eval_model.py
```
- Loads trained model
- Evaluates on test data
- Shows performance metrics and visualizations

### Complete Workflow Example

```bash
# 1. Set up virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# 2. Collect historical data
python main.py store_past_data --symbol BTCUSDT --days 7

# 3. Create labeled sequences (CSV)
python main.py label --symbol BTCUSDT

# 4. Train the model
python src/models/train_model.py

# 5. (Optional) Optimize hyperparameters
python src/models/tuning_optuna.py

# 6. Evaluate the model
python src/models/eval_model.py

# 7. Start the API server with dashboard
python main.py api --debug

# 8. Open dashboard in browser
# Go to: http://localhost:5000
```

### Command Examples

```bash
# Quick start - get last 7 days of BTCUSDT data
python main.py store_past_data

# Check how much data you have
python main.py status --symbol BTCUSDT

# Create CSV with custom filename
python main.py label --symbol BTC/USDT --output my_training_data.csv

# Start collecting real-time data with max 1000 iterations
python main.py collect --symbol BTCUSDT --max-iterations 1000

# Test all connections
python main.py test

# Start API server on custom port
python main.py api --port 8080 --debug

# Start API server on localhost only
python main.py api --host 127.0.0.1 --debug
```

### API Endpoints (when server is running)

```bash
# Dashboard UI
GET http://localhost:5000/

# Health check
GET http://localhost:5000/health

# Get prediction
GET http://localhost:5000/predict?symbol=BTCUSDT
POST http://localhost:5000/predict
{
  "symbol": "BTCUSDT"
}

# Get model metrics
GET http://localhost:5000/metrics

# Submit feedback
POST http://localhost:5000/feedback
{
  "symbol": "BTCUSDT",
  "actual_result": "UP",
  "timestamp": "2024-01-01T12:00:00Z"
}

# Bulk predictions
POST http://localhost:5000/bulk_predict
{
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
}
```

---

## 📊 Model Performance Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Accuracy** | 52.9% | Above random (50%) |
| **F1-Score** | 40.8% | Balanced precision/recall |
| **DOWN Recall** | 74% | Good at catching drops |
| **UP Recall** | 32% | Conservative on up moves |
| **Features** | 5,943 | Rich technical indicator set |
| **Training Data** | 820 sequences | 7 days of 1-min candles |

---

## 🎯 Key Achievements

- ✅ **End-to-end ML pipeline** from data collection to trained model
- ✅ **Production-ready architecture** with proper logging and error handling  
- ✅ **Hyperparameter optimization** using Optuna for best performance
- ✅ **Time-series validation** preventing data leakage
- ✅ **Comprehensive feature engineering** with 40+ technical indicators
- ✅ **Model persistence** and evaluation framework ready for deployment

Ready for **Phase 4: Flask API Integration** to serve live predictions! 🚀
We already have the Flask blueprint setup in `trading_api.py` with a `/predict` endpoint scaffold.
```