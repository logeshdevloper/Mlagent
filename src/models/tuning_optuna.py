import optuna
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, accuracy_score
import joblib
import json
from datetime import datetime

def objective(trial):
    """Optuna objective function to optimize LightGBM hyperparameters"""
    
    # Load your data
    df = pd.read_csv("data/firstdataset.csv")
    X = df.drop(columns=["label", "symbol", "created_at"])
    y = df["label"]
    
    # Suggest hyperparameters
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
        "is_unbalance": True,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }
    
    # Time series cross-validation
    cv = TimeSeriesSplit(n_splits=4)
    scores = []
    
    for train_idx, val_idx in cv.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        preds = model.predict(X_val_fold)
        score = accuracy_score(y_val_fold, preds)
        scores.append(score)
    
    return sum(scores) / len(scores)

if __name__ == "__main__":
    print("🔍 Starting Optuna hyperparameter optimization...")
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name="lightgbm_trading_optimization"
    )
    
    # Optimize
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save study object
    joblib.dump(study, f"artifacts/optuna_study_{timestamp}.pkl")
    
    # Save best parameters
    best_params = study.best_params
    best_params_file = f"artifacts/best_params_optuna_{timestamp}.json"
    with open(best_params_file, 'w') as f:
        json.dump(best_params, f, indent=2)
    
    print(f"\n✅ Optimization complete!")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best parameters saved to: {best_params_file}")
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
