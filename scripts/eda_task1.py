"""
EDA Script for Task 1: Insurance Data Analysis
Reads large data file in chunks, summarizes, and visualizes key insights.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File path
DATA_PATH = os.path.join('data', 'raw', 'MachineLearningRating_v3.txt')
CHUNKSIZE = 100_000  # Adjust as needed for memory

# Columns of interest for EDA
NUM_COLS = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
CAT_COLS = ['Province', 'VehicleType', 'Gender', 'CoverType', 'Make', 'Model']
DATE_COL = 'TransactionMonth'

# 1. Data type and missing value summary
def summarize_data():
    chunk = pd.read_csv(DATA_PATH, sep='\t', nrows=10)
    print('Column types:')
    print(chunk.dtypes)
    print('\nSample rows:')
    print(chunk.head())

# 2. Descriptive statistics and missing values
def describe_data():
    stats = {col: [] for col in NUM_COLS}
    na_counts = {col: 0 for col in NUM_COLS}
    total = 0
    for chunk in pd.read_csv(DATA_PATH, sep='\t', usecols=NUM_COLS, chunksize=CHUNKSIZE):
        for col in NUM_COLS:
            stats[col].append(chunk[col].describe())
            na_counts[col] += chunk[col].isna().sum()
        total += len(chunk)
    print('\nMissing values:')
    for col in NUM_COLS:
        print(f'{col}: {na_counts[col]} missing of {total}')

# 3. Plot distributions and outliers
def plot_distributions():
    for col in NUM_COLS:
        data = []
        for chunk in pd.read_csv(DATA_PATH, sep='\t', usecols=[col], chunksize=CHUNKSIZE):
            data.append(chunk[col].dropna())
        series = pd.concat(data)
        plt.figure(figsize=(8,4))
        sns.histplot(series, bins=50, kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(f'docs/{col}_hist.png')
        plt.close()
        # Boxplot for outliers
        plt.figure(figsize=(6,2))
        sns.boxplot(x=series)
        plt.title(f'Outliers in {col}')
        plt.savefig(f'docs/{col}_box.png')
        plt.close()

# 4. Loss Ratio by Province, VehicleType, Gender
def loss_ratio_by_group():
    group_cols = ['Province', 'VehicleType', 'Gender']
    for group in group_cols:
        agg = []
        for chunk in pd.read_csv(DATA_PATH, sep='\t', usecols=[group, 'TotalPremium', 'TotalClaims'], chunksize=CHUNKSIZE):
            agg.append(chunk.groupby(group)[['TotalPremium', 'TotalClaims']].sum())
        df = pd.concat(agg).groupby(level=0).sum()
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
        df = df.sort_values('LossRatio', ascending=False)
        plt.figure(figsize=(10,4))
        sns.barplot(x=df.index, y=df['LossRatio'])
        plt.title(f'Loss Ratio by {group}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'docs/loss_ratio_by_{group}.png')
        plt.close()

# 5. Temporal trends in claims and premium
def plot_temporal_trends():
    agg = []
    for chunk in pd.read_csv(DATA_PATH, sep='\t', usecols=[DATE_COL, 'TotalPremium', 'TotalClaims'], parse_dates=[DATE_COL], chunksize=CHUNKSIZE):
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors='coerce')
        agg.append(chunk.groupby(DATE_COL)[['TotalPremium', 'TotalClaims']].sum())
    df = pd.concat(agg).groupby(level=0).sum().sort_index()
    plt.figure(figsize=(12,5))
    df[['TotalPremium', 'TotalClaims']].plot()
    plt.title('Monthly Total Premium and Claims')
    plt.ylabel('Amount')
    plt.savefig('docs/monthly_trends.png')
    plt.close()

if __name__ == '__main__':
    summarize_data()
    describe_data()
    plot_distributions()
    loss_ratio_by_group()
    plot_temporal_trends()
    print('EDA complete. Plots saved in docs/.')
