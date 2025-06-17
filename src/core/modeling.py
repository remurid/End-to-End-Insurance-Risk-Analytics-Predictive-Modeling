import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

class ModelBuilder:
    def __init__(self, model_type='regression', algorithm='linear'):
        self.model_type = model_type
        self.algorithm = algorithm
        self.model = self._init_model()

    def _init_model(self):
        if self.model_type == 'regression':
            if self.algorithm == 'linear':
                return LinearRegression()
            elif self.algorithm == 'random_forest':
                return RandomForestRegressor(random_state=42)
            elif self.algorithm == 'xgboost':
                return XGBRegressor(random_state=42)
        elif self.model_type == 'classification':
            if self.algorithm == 'logistic':
                return LogisticRegression(max_iter=1000, random_state=42)
            elif self.algorithm == 'random_forest':
                return RandomForestClassifier(random_state=42)
            elif self.algorithm == 'xgboost':
                return XGBClassifier(random_state=42)
        raise ValueError('Unsupported model type or algorithm')

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        if self.model_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return {'RMSE': rmse, 'R2': r2}
        elif self.model_type == 'classification':
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}

    def get_model(self):
        return self.model

# SHAP and LIME for interpretability
import shap
import lime
import lime.lime_tabular

def shap_feature_importance(model, X, feature_names, top_n=10):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    # Return top features
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_features = np.argsort(mean_abs_shap)[::-1][:top_n]
    return [(feature_names[i], mean_abs_shap[i]) for i in top_features]

def lime_feature_importance(model, X, y, feature_names, top_n=10):
    explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=feature_names, class_names=['No', 'Yes'], discretize_continuous=True)
    exp = explainer.explain_instance(X.values[0], model.predict_proba if hasattr(model, 'predict_proba') else model.predict, num_features=top_n)
    return exp.as_list()
