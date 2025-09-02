import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import lightgbm as lgb

def load_and_evaluate():
    """Load saved model and perform comprehensive evaluation"""
    
    # Load model
    try:
        model = joblib.load("artifacts/model_v1_lgb.pkl")
        print("✅ Model loaded successfully")
    except FileNotFoundError:
        print("❌ Model not found. Run train_model.py first!")
        return
    
    # Load data
    df = pd.read_csv("data/firstdataset.csv")
    X = df.drop(columns=["label", "symbol", "created_at"])
    y = df["label"]
    
    # Use same split as training (last 15% as test set)
    n = len(df)
    train_size = int(n * 0.85)  # 70% train + 15% val = 85%
    
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    
    print(f"📊 Evaluating on {len(X_test)} test samples")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of UP
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"\n📈 Model Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Plot 2: Feature Importance (Top 20)
    plt.subplot(1, 3, 2)
    feature_names = X.columns
    importances = model.feature_importances_
    
    # Get top 20 features
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    
    plt.barh(range(20), feature_importance_df['importance'][::-1])
    plt.yticks(range(20), feature_importance_df['feature'][::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    
    # Plot 3: Prediction Probability Distribution
    plt.subplot(1, 3, 3)
    plt.hist(y_pred_proba[y_test == 0], alpha=0.7, label='Actual DOWN', bins=30)
    plt.hist(y_pred_proba[y_test == 1], alpha=0.7, label='Actual UP', bins=30)
    plt.xlabel('Predicted Probability of UP')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('artifacts/evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
    
    # Show top important features
    print(f"\n🎯 Top 10 Most Important Features:")
    for i, (feature, importance) in enumerate(feature_importance_df.head(10).values):
        print(f"  {i+1:2d}. {feature:<20}: {importance:.4f}")
    
    # Prediction confidence analysis
    confidence_high = (y_pred_proba > 0.7) | (y_pred_proba < 0.3)
    high_conf_accuracy = accuracy_score(y_test[confidence_high], y_pred[confidence_high])
    
    print(f"\n🎲 Confidence Analysis:")
    print(f"  High confidence predictions: {confidence_high.sum()}/{len(y_test)} ({confidence_high.mean():.1%})")
    print(f"  High confidence accuracy: {high_conf_accuracy:.4f}")
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'feature_importance': feature_importance_df
    }

if __name__ == "__main__":
    results = load_and_evaluate()
