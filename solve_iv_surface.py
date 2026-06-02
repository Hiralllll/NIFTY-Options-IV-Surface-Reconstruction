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

    for i, tlm in enumerate(tgt_lm):
        is_interior = slm[0] <= tlm <= slm[-1]

        if is_interior:
            # Akima interpolation (best for interior)
            if ak is not None:
                v = float(ak(tlm))
                if not np.isnan(v) and v > 0.005:
                    results[i] = v
                    continue

            # fallback: piecewise linear
            idx_b = np.searchsorted(slm, tlm) - 1
            idx_b = max(0, min(idx_b, len(slm) - 2))
            idx_a = idx_b + 1
            if slm[idx_a] != slm[idx_b]:
                w = (tlm - slm[idx_b]) / (slm[idx_a] - slm[idx_b])
                v = siv[idx_b] * (1 - w) + siv[idx_a] * w
            else:
                v = siv[idx_b]
            if v > 0.005:
                results[i] = v

        else:
            # Edge extrapolation: blend Akima derivative + 3-pt linear regression
            if tlm < slm[0]:
                if ak is not None:
                    try:
                        ak_slope = float(ak(slm[0], 1))
                    except:
                        ak_slope = None
                else:
                    ak_slope = None

                n_pts = min(3, len(slm))
                if n_pts >= 2:
                    c_lin = np.polyfit(slm[:n_pts], siv[:n_pts], 1)
                    lin_slope = c_lin[0]
                else:
                    lin_slope = 0.0

                if ak_slope is not None:
                    slope = 0.3 * ak_slope + 0.7 * lin_slope
                else:
                    slope = lin_slope

                dist = slm[0] - tlm
                v = siv[0] - slope * (1 - np.exp(-10.0 * dist)) / 10.0

            else:
                if ak is not None:
                    try:
                        ak_slope = float(ak(slm[-1], 1))
                    except:
                        ak_slope = None
                else:
                    ak_slope = None

                n_pts = min(3, len(slm))
                if n_pts >= 2:
                    c_lin = np.polyfit(slm[-n_pts:], siv[-n_pts:], 1)
                    lin_slope = c_lin[0]
                else:
                    lin_slope = 0.0

                if ak_slope is not None:
                    slope = 0.3 * ak_slope + 0.7 * lin_slope
                else:
                    slope = lin_slope

                dist = tlm - slm[-1]
                v = siv[-1] + slope * (1 - np.exp(-10.0 * dist)) / 10.0

            if v > 0.005:
                results[i] = v
            else:
                # clamp to nearest observed
                results[i] = siv[0] if tlm < slm[0] else siv[-1]

    return results


last_known = {c: np.nan for c in feature_cols}

for row_idx in range(len(df)):
    underlying = df.loc[row_idx, 'underlying_price']

    observed_mask = {c: not pd.isna(df.loc[row_idx, c]) for c in feature_cols}
    for c in feature_cols:
        if observed_mask[c]:
            last_known[c] = df.loc[row_idx, c]

    missing_ce = [c for c in ce_cols if not observed_mask[c]]
    missing_pe = [c for c in pe_cols if not observed_mask[c]]

    if not missing_ce and not missing_pe:
        continue

    for otype, cols, strikes_arr, missing in [
        ('CE', ce_cols, ce_strikes_arr, missing_ce),
        ('PE', pe_cols, pe_strikes_arr, missing_pe),
    ]:
        if not missing:
            continue

        obs_mask = [observed_mask[c] for c in cols]
        obs_strikes = strikes_arr[obs_mask]
        obs_ivs = np.array([df.loc[row_idx, c] for c, m in zip(cols, obs_mask) if m])
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
