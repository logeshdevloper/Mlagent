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

## 🚀 Current Status: Phase 3 Completed ✅

**Latest Achievement**: First production-ready ML model trained and evaluated
- **Model Accuracy**: 52.9% on test data
- **Model Type**: LightGBM Classifier
- **Dataset**: 820 sequences × 5,943 features
- **Performance**: Better at detecting DOWN moves (74% recall) vs UP moves (32% recall)

Next: **Phase 4 - FastAPI Integration**

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

### Phase 4 - Flask API Integration (Next)
- `/predict` endpoint for real-time predictions (already scaffolded in trading_api.py)
- `/feedback` endpoint for model improvement
- `/metrics` endpoint for performance tracking
- Live candle integration with existing Binance fetcher

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