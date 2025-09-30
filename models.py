"""
Machine Learning Models for Churn Prediction
Implements Random Forest and XGBoost classifiers
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             precision_recall_curve, roc_curve, f1_score)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Churn prediction model wrapper"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.metrics = {}
        
    def prepare_features(self, df, fit=True):
        """Prepare features for modeling"""
        logger.info("Preparing features...")
        
        numeric_features = [
            'video_duration_min', 'video_start_time_sec', 'buffering_count',
            'buffering_duration_sec', 'avg_bitrate_kbps', 'bitrate_switches',
            'watch_time_min', 'completion_rate', 'engagement_score',
            'buffering_ratio', 'quality_score', 'is_peak_hour', 
            'is_weekend', 'bitrate_stability'
        ]
        
        categorical_features = ['device', 'network_type', 'content_type', 
                               'resolution', 'isp']
        
        df_encoded = df.copy()
        for col in categorical_features:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df_encoded[col] = self.label_encoders[col].transform(df[col])
        
        feature_cols = numeric_features + categorical_features
        X = df_encoded[feature_cols]
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        self.feature_names = feature_cols
        return pd.DataFrame(X_scaled, columns=feature_cols)
    
    def build_model(self):
        """Build the ML model"""
        logger.info(f"Building {self.model_type} model...")
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                tree_method='hist'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        logger.info("Training model...")
        
        if self.model is None:
            self.build_model()
        
        if self.model_type == 'xgboost' and X_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        logger.info("Training completed")
        
        logger.info("Running cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc',
            n_jobs=-1
        )
        self.metrics['cv_auc_mean'] = cv_scores.mean()
        self.metrics['cv_auc_std'] = cv_scores.std()
        logger.info(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        self.metrics['test_auc'] = roc_auc_score(y_test, y_pred_proba)
        self.metrics['test_f1'] = f1_score(y_test, y_pred)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        self.metrics['precision'] = report['1']['precision']
        self.metrics['recall'] = report['1']['recall']
        
        logger.info(f"Test AUC: {self.metrics['test_auc']:.4f}")
        logger.info(f"Test F1: {self.metrics['test_f1']:.4f}")
        logger.info(f"Precision: {self.metrics['precision']:.4f}")
        logger.info(f"Recall: {self.metrics['recall']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm)
        self.plot_roc_curve(y_test, y_pred_proba)
        self.plot_feature_importance()
        
        return self.metrics
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Not Churned', 'Churned'],
                   yticklabels=['Not Churned', 'Churned'])
        plt.title(f'Confusion Matrix - {self.model_type}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'results/confusion_matrix_{self.model_type}.png', dpi=300)
        plt.close()
        logger.info(f"Confusion matrix saved")
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_type} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/roc_curve_{self.model_type}.png', dpi=300)
        plt.close()
        logger.info(f"ROC curve saved")
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), [self.feature_names[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importances - {self.model_type}')
            plt.tight_layout()
            plt.savefig(f'results/feature_importance_{self.model_type}.png', dpi=300)
            plt.close()
            logger.info(f"Feature importance plot saved")
    
    def save_model(self, path):
        """Save model and preprocessing objects"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
        
        metrics_path = path.replace('.pkl', '_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    @classmethod
    def load_model(cls, path):
        """Load saved model"""
        model_data = joblib.load(path)
        
        predictor = cls(model_type=model_data['model_type'])
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.label_encoders = model_data['label_encoders']
        predictor.feature_names = model_data['feature_names']
        predictor.metrics = model_data['metrics']
        
        logger.info(f"Model loaded from {path}")
        return predictor


def main():
    """Main training pipeline"""
    Path('results').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)
    
    logger.info("Loading data...")
    df = pd.read_csv('data/raw/streaming_sessions.csv')
    
    X = df.drop(['session_id', 'user_id', 'timestamp', 'churned', 'hour'], axis=1)
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    for model_type in ['random_forest', 'xgboost']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"{'='*50}")
        
        predictor = ChurnPredictor(model_type=model_type)
        
        X_train_prep = predictor.prepare_features(X_train, fit=True)
        X_val_prep = predictor.prepare_features(X_val, fit=False)
        X_test_prep = predictor.prepare_features(X_test, fit=False)
        
        predictor.train(X_train_prep, y_train, X_val_prep, y_val)
        
        metrics = predictor.evaluate(X_test_prep, y_test)
        
        predictor.save_model(f'models/{model_type}_model.pkl')
        
        print(f"\n{model_type.upper()} Results:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()