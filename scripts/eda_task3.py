import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/utils')))

import pandas as pd
from data_prep import load_data, preprocess_data
from metrics import compute_claim_frequency, compute_claim_severity, compute_margin
from segmentation import split_groups
from stat_tests import run_chi2_test, run_ttest
from reporting import interpret_result, business_recommendation

# Load and preprocess data
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/raw/MachineLearningRating_v3.txt'))
df = load_data(DATA_PATH)
df = preprocess_data(df)

# --- H₀: There are no risk differences across provinces ---
# Select metrics
province_freq = compute_claim_frequency(df, 'Province')
province_sev = compute_claim_severity(df, 'Province')
province_margin = compute_margin(df, 'Province')

# Data segmentation: Example with two provinces
prov_a, prov_b = 'Gauteng', 'Western Cape'
group_a, group_b = split_groups(df, 'Province', prov_a, prov_b)

# Statistical testing
freq_test = run_chi2_test(group_a, group_b, 'ClaimCount')
sev_test = run_ttest(group_a[group_a['ClaimCount']>0], group_b[group_b['ClaimCount']>0], 'ClaimAmount')
margin_test = run_ttest(group_a, group_b, 'TotalPremium')

# Analyze and report
print('--- Province Risk Differences ---')
print('Claim Frequency:', interpret_result(freq_test))
print('Claim Severity:', interpret_result(sev_test))
print('Margin:', interpret_result(margin_test))
print(business_recommendation('provinces', freq_test, prov_a, prov_b))

# --- H₀: There are no risk differences between zip codes ---
# Example: use two zip codes
zip_a, zip_b = df['ZipCode'].value_counts().index[:2]
group_a, group_b = split_groups(df, 'ZipCode', zip_a, zip_b)
freq_test = run_chi2_test(group_a, group_b, 'ClaimCount')
sev_test = run_ttest(group_a[group_a['ClaimCount']>0], group_b[group_b['ClaimCount']>0], 'ClaimAmount')
margin_test = run_ttest(group_a, group_b, 'TotalPremium')
print('\n--- Zip Code Risk Differences ---')
print('Claim Frequency:', interpret_result(freq_test))
print('Claim Severity:', interpret_result(sev_test))
print('Margin:', interpret_result(margin_test))
print(business_recommendation('zip codes', freq_test, zip_a, zip_b))

# --- H₀: There are no significant margin (profit) difference between zip codes ---
margin_test = run_ttest(group_a, group_b, 'TotalPremium')
print('\n--- Zip Code Margin Differences ---')
print('Margin:', interpret_result(margin_test))
print(business_recommendation('zip code margin', margin_test, zip_a, zip_b))

# --- H₀: There are not significant risk difference between Women and Men ---
group_a, group_b = split_groups(df, 'Gender', 'Female', 'Male')
freq_test = run_chi2_test(group_a, group_b, 'ClaimCount')
sev_test = run_ttest(group_a[group_a['ClaimCount']>0], group_b[group_b['ClaimCount']>0], 'ClaimAmount')
margin_test = run_ttest(group_a, group_b, 'TotalPremium')
print('\n--- Gender Risk Differences ---')
print('Claim Frequency:', interpret_result(freq_test))
print('Claim Severity:', interpret_result(sev_test))
print('Margin:', interpret_result(margin_test))
print(business_recommendation('gender', freq_test, 'Female', 'Male'))
