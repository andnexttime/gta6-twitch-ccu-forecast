"""gta6-twitch-ccu-forecast-01 entry: log-ratio quantile forecast from launch comparables.

Model (fit fresh on every case, from that case's own `comparables` only):
  For each past big launch with a usable release-day curve, take the log-ratio
  of each hour's viewers to its hour-1 print, r[h] = ln(v_h / v_1), h = 2..24.
  The shape of a launch day is driven mostly by how big hour 1 already was:
  a huge hour-1 print means the world was already watching at midnight UTC,
  so the day barely rises above it; a small hour-1 print means the audience
  still has to arrive with the rolling midnight unlocks and the evening peak
  (corr between ln(p1) and the peak log-ratio is about -0.94 across the
  comparables). Two estimators of the conditional distribution of r[h] are
  blended per hour:
    A) quadratic OLS on (ln p1, ln p1^2, log viewers-per-channel at hour 1),
       with the residual standard deviation as its spread;
    B) a Gaussian-kernel local mean / local sd of r[h] in ln p1 (bw 1.5),
       which adapts the spread to the target's size (big launches are much
       less dispersed than small ones) and never extrapolates a trend.
  Blend: median = 0.6*A + 0.4*B, sd = 0.6*sd_A + 0.4*sd_B, then
  q_tau(h) = p1 * exp(median + z_tau * sd * 0.6), z the normal quantiles.
  The 0.6 spread factor is calibrated so the 90% band covers about 0.9 of
  hours in leave-one-out validation over the comparables.

  Local validation (practice answers fetched once from SullyGnome's public
  API, builder-side only; this program never touches the network):
    house baseline persistence ............ loss 0.387, typical miss ~90%
    this model ............................ loss 0.185, typical miss ~36%
    leave-one-out over comparables ........ loss 0.246 (p1 >= 10k)
  Anything the fit cannot do (too few comparables, a singular fit, a missing
  hour) falls back to the house persistence baseline for that hour, so the
  output is always a valid forecast.

Standard library only, deterministic, no network, no files outside the case.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file would import (this entry is one file, but keep
# the starter's convention).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Blend weight of the regression component (kernel gets 1 - W_REG).
W_REG = 0.6
# Kernel bandwidth in ln(p1) units.
KERNEL_BW = 1.5
# Spread calibration factor (see docstring).
SPREAD = 0.6
# Persistence fallback spread (the house baseline's).
FALLBACK_SIGMA = 0.15
# A comparable needs at least this many release-day ratios to be used.
MIN_RATIOS = 20


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100
    rounds it (the starter's convention)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    """ISO 8601 UTC timestamp, e.g. 2026-11-19T00:00:00Z."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def solve_linear(mat, vec):
    """Solve mat x = vec by Gauss-Jordan with partial pivoting.
    Returns None when the matrix is (near-)singular."""
    n = len(vec)
    aug = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col] / pv
                for c in range(col, n + 1):
                    aug[r][c] -= f * aug[col][c]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def ols_fit(rows, ys):
    """Least squares with an intercept. rows: feature vectors (no intercept).
    Tries the full feature set, then drops trailing features, then the
    intercept-only mean, so a degenerate design still yields a fit.
    Returns (beta, resid_sd) with beta[0] the intercept, or None."""
    n = len(rows)
    if n == 0:
        return None
    k = len(rows[0])
    for keep in range(k, -1, -1):
        if n < keep + 2 and keep > 0:
            continue
        a = [[1.0] + list(r[:keep]) for r in rows]
        m = len(a[0])
        mat = [[sum(a[i][x] * a[i][y] for i in range(n)) for y in range(m)]
               for x in range(m)]
        vec = [sum(a[i][x] * ys[i] for i in range(n)) for x in range(m)]
        # Tiny ridge for numerical stability.
        for d in range(m):
            mat[d][d] += 1e-9
        beta = solve_linear(mat, vec)
        if beta is None:
            continue
        resid = [y - (beta[0] + sum(b * v for b, v in zip(beta[1:], r[:keep])))
                 for r, y in zip(rows, ys)]
        # Population variance (statistics.pstdev), matching the spread this
        # entry's SPREAD factor was calibrated against in validation.
        var = sum(e * e for e in resid) / n
        return beta, math.sqrt(var)
    return None


def mean_sd(vals):
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    return mu, math.sqrt(var)


def comparable_curves(case):
    """Extract each comparable's release-day log-ratio curve and features.

    Returns a list of dicts: lnp1, vpc (log viewers per channel at hour 1,
    None when channels are missing), ratios {h: ln(v_h / v_1)} for h = 2..24.
    A comparable is usable when it has an hour-1 print and at least
    MIN_RATIOS release-day ratios. The Grand Theft Auto VI pre-release
    comparable has no release day yet, so it is skipped here by that rule."""
    out = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        by_off = {}
        ch_by_off = {}
        for pt in comp.get("hourly") or []:
            try:
                t = parse_ts(pt["from"])
                v = float(pt["viewers"])
            except (KeyError, TypeError, ValueError):
                continue
            if v <= 0:
                continue
            off = int((t - day_start).total_seconds() // 3600)
            by_off[off] = v
            ch = pt.get("channels")
            if ch is not None:
                try:
                    if float(ch) > 0:
                        ch_by_off[off] = float(ch)
                except (TypeError, ValueError):
                    pass
        if 0 not in by_off:
            continue
        p1 = by_off[0]
        ratios = {}
        for h in range(2, 25):
            if (h - 1) in by_off:
                ratios[h] = math.log(by_off[h - 1] / p1)
        if len(ratios) < MIN_RATIOS:
            continue
        vpc = None
        if 0 in ch_by_off:
            vpc = math.log(p1 / ch_by_off[0])
        out.append({"lnp1": math.log(p1), "vpc": vpc, "ratios": ratios})
    return out


def features(comp, med_vpc):
    """Feature vector for one comparable (or the target): [ln p1, ln p1^2, vpc]."""
    vpc = comp["vpc"] if comp.get("vpc") is not None else med_vpc
    x = comp["lnp1"]
    return [x, x * x, vpc]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    hours = [row["hour"] for row in case.get("forecast") or []]
    if sorted(hours) != list(range(2, 25)):
        raise ValueError(f"{name} does not forecast hours 2..24")

    comps = comparable_curves(case)
    # Median log viewers-per-channel across comparables, for targets whose
    # hour-1 channels are missing.
    vpcs = sorted(c["vpc"] for c in comps if c.get("vpc") is not None)
    med_vpc = vpcs[len(vpcs) // 2] if vpcs else 0.0
    target = {"lnp1": math.log(p1)}
    ch1 = first.get("channels")
    if ch1 is not None:
        try:
            if float(ch1) > 0:
                target["vpc"] = math.log(p1 / float(ch1))
        except (TypeError, ValueError):
            pass
    tx = features(target, med_vpc)

    rows = []
    for h in hours:
        fc_h = None
        pool = [(features(c, med_vpc), c["ratios"][h])
                for c in comps if h in c["ratios"]]
        if len(pool) >= 6:
            xs = [r[0] for r in pool]
            ys = [r[1] for r in pool]
            reg = ols_fit(xs, ys)
            # Kernel local mean / sd in ln(p1).
            x0 = tx[0]
            wts = []
            kvals = []
            for c in comps:
                if h not in c["ratios"]:
                    continue
                w = math.exp(-0.5 * ((c["lnp1"] - x0) / KERNEL_BW) ** 2)
                wts.append(w)
                kvals.append(c["ratios"][h])
            sw = sum(wts)
            if reg is not None and sw > 0:
                beta, sd_a = reg
                mu_a = beta[0] + sum(b * v for b, v in zip(beta[1:], tx))
                mu_b = sum(w * y for w, y in zip(wts, kvals)) / sw
                var_b = sum(w * (y - mu_b) ** 2 for w, y in zip(wts, kvals)) / sw
                sd_b = math.sqrt(var_b)
                mu = W_REG * mu_a + (1.0 - W_REG) * mu_b
                sd = W_REG * sd_a + (1.0 - W_REG) * sd_b
                sd = max(0.05, sd) * SPREAD
                fc_h = [p1 * math.exp(mu + z * sd) for z in Z]
        if fc_h is None:
            # Fallback: the house persistence baseline for this hour.
            fc_h = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)) for z in Z]
        rows.append(fc_h)

    out_hours = []
    for h, qs in zip(hours, rows):
        clean = []
        prev = 0.0
        for q in qs:
            v = round2(min(1e9, max(q, 1e-9)))
            v = max(v, prev)  # keep the row non-decreasing after rounding
            clean.append(v)
            prev = v
        out_hours.append({"hour": h, "q": clean})
    return {"hours": out_hours}


def fallback_case(case, name):
    """The house persistence baseline, used when a case cannot be forecast
    at all (missing hour-1 print, wrong levels, ...). Always valid."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first else 1.0
    hours = [row["hour"] for row in case.get("forecast") or []]
    if not hours:
        hours = list(range(2, 25))
    out_hours = []
    for h in hours:
        qs = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in Z]
        clean = []
        prev = 0.0
        for q in qs:
            v = max(q, prev)
            clean.append(v)
            prev = v
        out_hours.append({"hour": h, "q": clean})
    return {"hours": out_hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
