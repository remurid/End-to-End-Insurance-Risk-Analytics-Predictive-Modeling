import unittest
import numpy as np
import pandas as pd
from src.core.modeling import ModelBuilder

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        # Regression data
        self.X_reg = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [5, 4, 3, 2, 1]
        })
        self.y_reg = np.array([2, 4, 6, 8, 10])
        # Classification data
        self.X_clf = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5, 6],
            'f2': [5, 4, 3, 2, 1, 0]
        })
        self.y_clf = np.array([0, 0, 1, 1, 0, 1])

    def test_linear_regression(self):
        model = ModelBuilder(model_type='regression', algorithm='linear')
        model.fit(self.X_reg, self.y_reg)
        metrics = model.evaluate(self.X_reg, self.y_reg)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)

    def test_random_forest_regression(self):
        model = ModelBuilder(model_type='regression', algorithm='random_forest')
        model.fit(self.X_reg, self.y_reg)
        metrics = model.evaluate(self.X_reg, self.y_reg)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)

    def test_xgboost_regression(self):
        model = ModelBuilder(model_type='regression', algorithm='xgboost')
        model.fit(self.X_reg, self.y_reg)
        metrics = model.evaluate(self.X_reg, self.y_reg)
        self.assertIn('RMSE', metrics)
        self.assertIn('R2', metrics)

    def test_logistic_regression(self):
        model = ModelBuilder(model_type='classification', algorithm='logistic')
        model.fit(self.X_clf, self.y_clf)
        metrics = model.evaluate(self.X_clf, self.y_clf)
        self.assertIn('Accuracy', metrics)
        self.assertIn('F1', metrics)

    def test_random_forest_classification(self):
        model = ModelBuilder(model_type='classification', algorithm='random_forest')
        model.fit(self.X_clf, self.y_clf)
        metrics = model.evaluate(self.X_clf, self.y_clf)
        self.assertIn('Accuracy', metrics)
        self.assertIn('F1', metrics)

    def test_xgboost_classification(self):
        model = ModelBuilder(model_type='classification', algorithm='xgboost')
        model.fit(self.X_clf, self.y_clf)
        metrics = model.evaluate(self.X_clf, self.y_clf)
        self.assertIn('Accuracy', metrics)
        self.assertIn('F1', metrics)

if __name__ == '__main__':
    unittest.main()
