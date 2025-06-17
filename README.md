# End-to-End Insurance Risk Analytics & Predictive Modeling

This repository contains code and analysis for insurance risk analytics and predictive modeling, including EDA, statistical modeling, and A/B testing.

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- Run EDA script:

  ```powershell
  python scripts/eda_task1.py
  ```

- Or open and run the notebook in `notebooks/eda_results.ipynb` for interactive EDA and visualization.

- **Statistical Hypothesis Testing:**

  Open and run the notebook in `notebooks/insurance_risk_hypothesis_testing.ipynb` to perform modular, statistical validation of key risk driver hypotheses (province, zip code, gender, margin, etc.) using the utility scripts in `src/utils/`.

## Project Structure

- `data/raw/`: Raw data files
- `notebooks/`: Jupyter notebooks for analysis
- `scripts/`: Python scripts for automation and EDA
- `src/utils/`: Modular utility scripts for data prep, metrics, segmentation, statistical tests, and reporting
- `src/`: Source code for modeling and utilities
- `tests/`: Unit and integration tests

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy

## Modular Utility Scripts

- `src/utils/data_prep.py`: Data loading and preprocessing
- `src/utils/metrics.py`: KPI computation (claim frequency, severity, margin)
- `src/utils/segmentation.py`: Data segmentation for A/B testing
- `src/utils/stat_tests.py`: Statistical test functions (chi-squared, t-test)
- `src/utils/reporting.py`: Result interpretation and business recommendation

