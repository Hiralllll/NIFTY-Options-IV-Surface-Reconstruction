import pandas as pd
import numpy as np
import re
from scipy.interpolate import Akima1DInterpolator

INPUT_FILE = "dataset.csv"
df = pd.read_csv(INPUT_FILE)

feature_cols = [c for c in df.columns if c not in ['datetime', 'underlying_price']]
col_info = {}
ce_cols, pe_cols = [], []

for c in feature_cols:
    m = re.match(r'NIFTY27JAN26(\d+)(CE|PE)', c)
    strike = int(m.group(1))
    otype = m.group(2)
    col_info[c] = {'strike': strike, 'type': otype}
    (ce_cols if otype == 'CE' else pe_cols).append(c)

ce_cols.sort(key=lambda c: col_info[c]['strike'])
pe_cols.sort(key=lambda c: col_info[c]['strike'])
ce_strikes_arr = np.array([col_info[c]['strike'] for c in ce_cols], dtype=float)
pe_strikes_arr = np.array([col_info[c]['strike'] for c in pe_cols], dtype=float)

from test_solve import interpolate_smile

total_mse = 0.0
count = 0
edge_mse = 0.0
edge_count = 0
interior_mse = 0.0
interior_count = 0
high_iv_mse = 0.0
high_iv_count = 0
low_iv_mse = 0.0
low_iv_count = 0

for row_idx in range(len(df)):
    if row_idx % 1000 == 0 and row_idx > 0:
        print(f"Row {row_idx}/{len(df)}...")
        
    underlying = df.loc[row_idx, 'underlying_price']
    observed_mask = {c: not pd.isna(df.loc[row_idx, c]) for c in feature_cols}
    
    for otype, cols, strikes_arr in [('CE', ce_cols, ce_strikes_arr), ('PE', pe_cols, pe_strikes_arr)]:
        obs_mask = [observed_mask[c] for c in cols]
        obs_cols = [c for c, m in zip(cols, obs_mask) if m]
        
        if len(obs_cols) < 4:
            continue
            
        for hide_idx, hide_col in enumerate(obs_cols):
            target_strike = np.array([col_info[hide_col]['strike']])
            true_iv = df.loc[row_idx, hide_col]
            
            train_cols = obs_cols[:hide_idx] + obs_cols[hide_idx+1:]
            train_strikes = np.array([col_info[c]['strike'] for c in train_cols])
            train_ivs = np.array([df.loc[row_idx, c] for c in train_cols])

            estimated = interpolate_smile(target_strike, train_strikes, train_ivs, underlying)
            pred_iv = estimated[0]
            
            if not np.isnan(pred_iv):
                sq_err = (pred_iv - true_iv) ** 2
                total_mse += sq_err
                count += 1
                
                if np.median(train_ivs) > 0.5:
                    high_iv_mse += sq_err
                    high_iv_count += 1
                else:
                    low_iv_mse += sq_err
                    low_iv_count += 1
                
                hide_strike = target_strike[0]
                if hide_strike < train_strikes[0] or hide_strike > train_strikes[-1]:
                    edge_mse += sq_err
                    edge_count += 1
                else:
                    interior_mse += sq_err
                    interior_count += 1

print("--- LOOCV Results ---")
print(f"Overall MSE:  {total_mse / count if count else 0:.8f} (N={count})")
print(f"Interior MSE: {interior_mse / interior_count if interior_count else 0:.8f} (N={interior_count})")
print(f"Edge MSE:     {edge_mse / edge_count if edge_count else 0:.8f} (N={edge_count})")
print(f"High IV MSE:  {high_iv_mse / high_iv_count if high_iv_count else 0:.8f} (N={high_iv_count})")
print(f"Low IV MSE:   {low_iv_mse / low_iv_count if low_iv_count else 0:.8f} (N={low_iv_count})")
