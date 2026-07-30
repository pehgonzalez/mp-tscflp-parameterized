#!/usr/bin/env python3
"""
Q3 analysis: rank association of the covering bound k* with the terminal gap
on the Mauri benchmark, pooled and within size groups.

Run this AFTER the benchmark MIP campaign (run_mip_mauri.ps1) finishes. It joins
the per-instance covering bound k* (from mauri_kstar.csv, computed in linear time
without solving anything) with the solver's terminal gap and time, and
reports the Spearman rank correlation pooled and within each size group, plus a
size-controlled (within-group averaged) coefficient. It then prints a LaTeX
sentence summarising the finding, for cross-checking against the paper.

Usage:
    python3 analyze_q3_correlation.py [mauri_mip.csv]

The MIP results CSV needs, per instance, at least: an instance-name column, a
solve-time column and a terminal-gap column. Column names are matched loosely
(case-insensitive, common aliases), so the campaign's raw output usually works
as-is. A status/timeout column, if present, is used to flag censoring.
"""
import csv, sys, os, math

KSTAR_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "mauri_kstar.csv")

# ---- loose column matching ------------------------------------------------
INSTANCE_KEYS = ("instance", "name", "file", "instance_name")
TIME_KEYS     = ("solver_time_s", "total_wall_s", "time_s", "time",
                 "solve_time", "runtime", "seconds", "wall_s")
GAP_KEYS      = ("gap", "mipgap", "terminal_gap", "rel_gap", "gap_pct")
STATUS_KEYS   = ("status", "termination", "timeout", "solved")

def pick(fieldnames, keys):
    low = {f.lower(): f for f in fieldnames}
    for k in keys:
        if k in low:
            return low[k]
    return None

def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

# ---- Spearman (no SciPy dependency) ---------------------------------------
def rankdata(a):
    order = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((v - mx) ** 2 for v in x))
    sy = math.sqrt(sum((v - my) ** 2 for v in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (sx * sy)

def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))

def group_label(row):
    # size group = |I|-|J|-|K| crossed with |L|, exactly as the paper groups them
    return f"{row['nI']}-{row['nJ']}-{row['nK']}-L{row['nL']}"

def main():
    mip_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "mauri_mip.csv")
    if not os.path.exists(mip_path):
        sys.exit(f"MIP results not found: {mip_path}\n"
                 "Run the benchmark campaign first, then pass its CSV path.")

    kstar = {r["instance"]: r for r in load_csv(KSTAR_CSV)}
    mip_rows = load_csv(mip_path)
    if not mip_rows:
        sys.exit(f"{mip_path} has no data rows.")
    fn = mip_rows[0].keys()
    ci, ct, cg = (pick(fn, INSTANCE_KEYS), pick(fn, TIME_KEYS), pick(fn, GAP_KEYS))
    cs = pick(fn, STATUS_KEYS)
    if not (ci and ct and cg):
        sys.exit(f"Could not locate instance/time/gap columns in {list(fn)}")

    rows = []
    for r in mip_rows:
        name = r[ci].strip()
        kr = kstar.get(name) or kstar.get(name + ".txt") or kstar.get(
            name.replace(".txt", ""))
        if not kr:
            print(f"  [skip] no k* for {name}")
            continue
        try:
            t = float(r[ct]); g = float(r[cg])
        except ValueError:
            continue
        rows.append({
            "instance": name, "kstar": float(kr["kstar_py"]),
            "time": t, "gap": g, "group": group_label(kr),
            "censored": (cs and str(r[cs]).strip().upper() in
                         ("TIMEOUT", "TIME_LIMIT", "FEASIBLE", "SUBOPTIMAL",
                          "1", "TRUE")),
        })

    if len(rows) < 3:
        sys.exit("Too few joined rows to correlate.")

    ks = [r["kstar"] for r in rows]
    tt = [r["time"] for r in rows]
    gg = [r["gap"] for r in rows]
    n_cens = sum(1 for r in rows if r["censored"])

    rho_t = spearman(ks, tt)
    rho_g = spearman(ks, gg)

    print(f"\nJoined {len(rows)} instances ({n_cens} censored at the time limit).")
    print(f"Pooled Spearman  k* vs time : rho = {rho_t:+.3f}")
    print(f"Pooled Spearman  k* vs gap  : rho = {rho_g:+.3f}")

    # within-group (controls for instance size, the confounder the paper flags)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    print("\nWithin-group (size held fixed):")
    wt, wg = [], []
    for gname, gr in sorted(groups.items()):
        if len(gr) < 3:
            print(f"  {gname:>16}: n={len(gr)} (too few)")
            continue
        rt = spearman([r["kstar"] for r in gr], [r["time"] for r in gr])
        rgap = spearman([r["kstar"] for r in gr], [r["gap"] for r in gr])
        wt.append(rt); wg.append(rgap)
        print(f"  {gname:>16}: n={len(gr):>3}  rho(time)={rt:+.3f}  rho(gap)={rgap:+.3f}")
    mwt = sum(wt) / len(wt) if wt else float("nan")
    mwg = sum(wg) / len(wg) if wg else float("nan")
    print(f"\nMean within-group rho  k* vs time : {mwt:+.3f}")
    print(f"Mean within-group rho  k* vs gap  : {mwg:+.3f}")

    # ---- LaTeX summary sentence (data-driven) ----------------------------
    print("\n" + "=" * 68)
    print("LaTeX summary sentence for cross-checking:\n")
    if math.isnan(rho_g):
        print("[gap correlation undefined -- inspect the CSV]")
    else:
        time_clause = "" if math.isnan(rho_t) else f" (time $\\rho={rho_t:+.2f}$)"
        wclause = "" if math.isnan(mwg) else (
            f", with mean within-group $\\rho={mwg:+.2f}$ on the gap")
        if rho_g <= -0.3:
            # larger k* -> smaller gap (easier): the inverse direction of Q1
            reading = (" so a larger covering bound goes with a smaller terminal "
                       "gap, the same inverse direction the enumeration shows in "
                       "Q1, tighter capacities making the instance easier")
        elif rho_g >= 0.3:
            reading = (" so a larger covering bound goes with more solver effort, "
                       "and the bound tracks difficulty beyond instance size")
        else:
            reading = (" so on this benchmark the bound is at best a weak monotone "
                       "predictor of solver effort")
        print(f"Over the $100$ benchmark instances the covering bound and the "
              f"\\textsc{{Mip}} terminal gap are rank-correlated with Spearman "
              f"$\\rho={rho_g:+.2f}${time_clause}{wclause},{reading}.")
    print("=" * 68)

if __name__ == "__main__":
    main()
