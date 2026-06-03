import pandas as pd
import numpy as np
from scipy.interpolate import Akima1DInterpolator
import re
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = "dataset.csv"
OUTPUT_FILLED = "filled_dataset.csv"
OUTPUT_SUBMISSION = "submission.csv"
SEPARATOR = "||"

df = pd.read_csv(INPUT_FILE)
filled = df.copy()

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


def interpolate_smile(target_strikes, obs_strikes, obs_ivs, underlying):
    n_obs = len(obs_strikes)
    results = np.full(len(target_strikes), np.nan)

    if n_obs == 0:
        return results
    if n_obs == 1:
        results[:] = obs_ivs[0]
        return results

    obs_lm = np.log(obs_strikes / underlying)
    tgt_lm = np.log(target_strikes / underlying)

    order = np.argsort(obs_lm)
    slm = obs_lm[order]
    siv = obs_ivs[order]

    # Akima for interior points (4+ observations needed)
    ak = None
    if n_obs >= 4:
        try:
            ak = Akima1DInterpolator(slm, siv)
        except Exception:
            pass

    # Quadratic polynomial if 3+ points
    quad_coeff = None
    if n_obs >= 3:
        try:
            quad_coeff = np.polyfit(slm, siv, 2)
            if quad_coeff[0] < 0:  # Prevent concave (upside-down U) fits from crashing edge extrapolations
                quad_coeff = None
        except Exception:
            pass

    decay_rate = 5.0  # TUNE ME: Monotonic Exponential Slope Decay rate

    for i, tlm in enumerate(tgt_lm):
        is_interior = slm[0] <= tlm <= slm[-1]

        if is_interior:
            # Akima interpolation (best for interior)
            if ak is not None:
                v = float(ak(tlm))
                if not np.isnan(v):
                    results[i] = max(v, 0.001)
                    continue

            # Quadratic fallback
            if quad_coeff is not None:
                v = float(np.polyval(quad_coeff, tlm))
                if not np.isnan(v):
                    results[i] = max(v, 0.001)
                    continue

            # fallback: piecewise linear (only if Akima and Quad completely failed)
            idx_b = np.searchsorted(slm, tlm) - 1
            idx_b = max(0, min(idx_b, len(slm) - 2))
            idx_a = idx_b + 1
            if slm[idx_a] != slm[idx_b]:
                w = (tlm - slm[idx_b]) / (slm[idx_a] - slm[idx_b])
                v = siv[idx_b] * (1 - w) + siv[idx_a] * w
            else:
                v = siv[idx_b]
                
            results[i] = max(v, 0.001)

        else:
            # Edge extrapolation: blend Akima/Quad derivative + 3-pt linear regression
            if tlm < slm[0]:
                ak_slope = None
                if ak is not None:
                    try:
                        ak_slope = float(ak(slm[0], 1))
                    except:
                        pass

                n_pts = min(3, len(slm))
                if n_pts >= 2:
                    c_lin = np.polyfit(slm[:n_pts], siv[:n_pts], 1)
                    lin_slope = c_lin[0]
                else:
                    lin_slope = 0.0

                if ak_slope is not None:
                    slope = 0.3 * ak_slope + 0.7 * lin_slope
                elif quad_coeff is not None:
                    slope = 2 * quad_coeff[0] * slm[0] + quad_coeff[1]
                else:
                    slope = lin_slope

                dist = slm[0] - tlm
                v = siv[0] - slope * (1 - np.exp(-decay_rate * dist)) / decay_rate

            else:
                ak_slope = None
                if ak is not None:
                    try:
                        ak_slope = float(ak(slm[-1], 1))
                    except:
                        pass

                n_pts = min(3, len(slm))
                if n_pts >= 2:
                    c_lin = np.polyfit(slm[-n_pts:], siv[-n_pts:], 1)
                    lin_slope = c_lin[0]
                else:
                    lin_slope = 0.0

                if ak_slope is not None:
                    slope = 0.3 * ak_slope + 0.7 * lin_slope
                elif quad_coeff is not None:
                    slope = 2 * quad_coeff[0] * slm[-1] + quad_coeff[1]
                else:
                    slope = lin_slope

                dist = tlm - slm[-1]
                v = siv[-1] + slope * (1 - np.exp(-decay_rate * dist)) / decay_rate

            results[i] = max(v, 0.001)

    return results


last_known = {c: np.nan for c in feature_cols}

for row_idx in range(len(df)):
    underlying = df.loc[row_idx, 'underlying_price']

    observed_mask = {c: not pd.isna(df.loc[row_idx, c]) for c in feature_cols}
    missing_ce = [c for c in ce_cols if not observed_mask[c]]
    missing_pe = [c for c in pe_cols if not observed_mask[c]]

    # Removed early continue to ensure last_known gets updated even if no missing values exist

    for otype, cols, strikes_arr, missing in [
        ('CE', ce_cols, ce_strikes_arr, missing_ce),
        ('PE', pe_cols, pe_strikes_arr, missing_pe),
    ]:
        if not missing:
            continue

        obs_mask = [observed_mask[c] for c in cols]
        obs_cols = [c for c, m in zip(cols, obs_mask) if m]
        obs_strikes = strikes_arr[obs_mask]
        obs_ivs = np.array([df.loc[row_idx, c] for c in obs_cols])
        miss_strikes = np.array([col_info[c]['strike'] for c in missing], dtype=float)

        estimated = (
            interpolate_smile(miss_strikes, obs_strikes, obs_ivs, underlying)
            if len(obs_strikes) >= 2
            else np.full(len(missing), np.nan)
        )

        for i, c in enumerate(missing):
            if not np.isnan(estimated[i]):
                filled.loc[row_idx, c] = estimated[i]
            elif not np.isnan(last_known[c]):
                filled.loc[row_idx, c] = last_known[c]
            elif len(obs_ivs) > 0:
                filled.loc[row_idx, c] = np.median(obs_ivs)
            else:
                all_last = [last_known[cc] for cc in cols if not np.isnan(last_known[cc])]
                filled.loc[row_idx, c] = np.median(all_last) if all_last else 0.12

    # CRITICAL FIX: Update last_known AT THE END of the row
    # using the FILLED values. This ensures that options missing for long periods
    # carry forward the most recent structurally interpolated smile, rather than
    # being stuck with days-old stale IVs!
    for c in feature_cols:
        val = filled.loc[row_idx, c]
        if not pd.isna(val):
            last_known[c] = val

remaining = filled[feature_cols].isna().sum().sum()
assert remaining == 0

for c in feature_cols:
    filled[c] = filled[c].clip(lower=0.001)

filled.to_csv(OUTPUT_FILLED, index=False)

original = pd.read_csv(INPUT_FILE)
rows = []
for col in feature_cols:
    was_missing = original[col].isna()
    for idx in original.index[was_missing]:
        dt = original.loc[idx, "datetime"]
        uid = f"{dt}{SEPARATOR}{col}"
        rows.append({"id": uid, "value": filled.loc[idx, col]})

solution = pd.DataFrame(rows, columns=["id", "value"])
solution = solution.sort_values("id").reset_index(drop=True)
solution.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Done — {len(solution)} rows written to {OUTPUT_SUBMISSION}")
