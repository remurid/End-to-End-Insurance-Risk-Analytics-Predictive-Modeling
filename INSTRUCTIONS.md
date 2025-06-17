# Instructions for Testing and Running the Predictive Modeling Pipeline

## 1. Ensure the Correct Python Environment
- In VS Code, open the Command Palette (`Ctrl+Shift+P`)
- Type and select `Python: Select Interpreter`
- Choose the interpreter located at:
  `./.venv/Scripts/python.exe`

## 2. Install Required Packages
If you encounter import errors for `xgboost`, `shap`, or `lime`, open a terminal in VS Code and run:

```
.venv\Scripts\pip install xgboost shap lime
```

## 3. Run Unit Tests
To verify the data preparation and modeling modules:

```
.venv\Scripts\python -m unittest tests/unit/test_data_prep.py
.venv\Scripts\python -m unittest tests/unit/test_modeling.py
```

Both should report all tests as passed.

## 4. Using the Modules in Your Workflow
- Use `src/utils/data_prep.py` for data loading, cleaning, feature engineering, encoding, and splitting.
- Use `src/core/modeling.py` to build, train, evaluate, and interpret models (regression and classification).

## 5. Model Interpretability
- For SHAP or LIME analysis, use the `shap_feature_importance` or `lime_feature_importance` functions in `modeling.py` after fitting your model.
- Example usage:

```python
from src.core.modeling import ModelBuilder, shap_feature_importance
model = ModelBuilder(model_type='regression', algorithm='xgboost')
model.fit(X_train, y_train)
shap_feature_importance(model.get_model(), X_train, X_train.columns, top_n=10)
```

## 6. Troubleshooting
- If you see `ModuleNotFoundError`, repeat step 2 and ensure the correct interpreter is selected.
- Restart VS Code after installing new packages if issues persist.

---

**You are now ready to run, test, and extend your end-to-end insurance risk analytics and predictive modeling pipeline!**
