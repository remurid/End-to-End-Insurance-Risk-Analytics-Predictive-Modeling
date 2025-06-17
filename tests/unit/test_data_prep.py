import unittest
import pandas as pd
from src.utils import data_prep

class TestDataPrep(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['x', 'y', 'x', None],
            'TotalClaims': [100, 200, 0, 400],
            'CalculatedPremiumPerTerm': [1000, 2000, 3000, 4000]
        })

    def test_handle_missing_data(self):
        df_filled = data_prep.handle_missing_data(self.df.copy(), strategy='mean', columns=['A'])
        self.assertFalse(df_filled['A'].isnull().any())

    def test_encode_categorical(self):
        df_encoded = data_prep.encode_categorical(self.df.copy(), columns=['B'], method='onehot')
        self.assertTrue(any('B_' in col for col in df_encoded.columns))

    def test_feature_engineering(self):
        df_feat = data_prep.feature_engineering(self.df.copy())
        self.assertIn('ClaimRatio', df_feat.columns)

    def test_split_data(self):
        df_clean = self.df.dropna()
        X_train, X_test, y_train, y_test = data_prep.split_data(df_clean, target='TotalClaims', test_size=0.5, random_state=1)
        self.assertEqual(len(X_train) + len(X_test), len(df_clean))

if __name__ == '__main__':
    unittest.main()
