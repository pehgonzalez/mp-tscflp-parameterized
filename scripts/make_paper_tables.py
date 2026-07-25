#!/usr/bin/env python3
"""
Gera as tabelas do corpo do artigo a partir dos CSVs brutos em results/.
Cada tabela publicada e' regenerada por este script, entao os numeros do
texto ficam pinados nos dados (diretriz 6 do playbook). Saida em
paper/tables/*.tex, incluidas por \\input em experiments.tex.

  tab_q1_boundary.tex  extensao de fronteira do Q1 (resolvidas e mediana por n x k*)
  tab_q2_codes.tex     comparacao dos tres codigos na familia controlada
  tab_q3_groups.tex    resumo por grupo do benchmark (k*, gap, tempo, otimos, rho)

Uso: python3 scripts/make_paper_tables.py
"""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES  = os.path.join(ROOT, "results")
OUT  = os.path.join(ROOT, "paper", "tables")
os.makedirs(OUT, exist_ok=True)

def load(name):
    with open(os.path.join(RES, name), newline="") as fh:
        return list(csv.DictReader(fh))

def median(xs):
    s = sorted(xs); n = len(s)
    if n == 0: return float("nan")
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])

def rankdata(a):
    order = sorted(range(len(a)), key=lambda i: a[i]); r=[0.0]*len(a); i=0
    while i < len(a):
        j = i
        while j+1 < len(a) and a[order[j+1]] == a[order[i]]: j += 1
        avg = (i+j)/2.0+1.0
        for k in range(i, j+1): r[order[k]] = avg
        i = j+1
    return r

def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y); n=len(x)
    mx, my = sum(rx)/n, sum(ry)/n
    sx = math.sqrt(sum((v-mx)**2 for v in rx)); sy = math.sqrt(sum((v-my)**2 for v in ry))
    return sum((rx[i]-mx)*(ry[i]-my) for i in range(n))/(sx*sy)

def perm_p(x, y, nperm=20000, seed=20260718):
    """p-valor bilateral por permutacao do Spearman com mid-ranks.
    Semente fixa, entao o numero publicado e' deterministico."""
    import random as _rnd
    rng = _rnd.Random(seed)
    obs = spearman(x, y); cnt = 0; y2 = list(y)
    for _ in range(nperm):
        rng.shuffle(y2)
        if abs(spearman(x, y2)) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (nperm + 1)

def boot_ci(x, y, nboot=10000, seed=20260719):
    """IC percentil bootstrap 95% do Spearman, semente fixa (numero pinado)."""
    import random as _rnd
    rng = _rnd.Random(seed)
    n = len(x); vals = []
    for _ in range(nboot):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [x[i] for i in idx]; by = [y[i] for i in idx]
        if len(set(bx)) < 2 or len(set(by)) < 2:
            continue
        vals.append(spearman(bx, by))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return lo, hi

def round_half_up(v, nd=2):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(v)).quantize(Decimal("0."+"0"*nd), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------- Q1 boundary
q1 = load("q1_xp.csv")
# extensao de fronteira 60 s: os 60 runs com nK=30 e nL=3 (o grid principal usa
# nK em {50,100} e nL em {5,10}); familia do orcamento 600 s: 10 sementes por
# celula sobre a mesma grade (results/q1_boundary600.csv)
bnd = [r for r in q1 if r["nK"] == "30" and r["nL"] == "3"]
b600 = load("q1_boundary600.csv")
ns   = sorted({int(r["nI"])+int(r["nJ"]) for r in bnd} |
              {int(r["nI"])+int(r["nJ"]) for r in b600})
kts  = [4, 8, 12, 16]
def cell_tex(cell, solved):
    if not cell:
        return "--"
    if solved:
        med = median([float(r["time"]) for r in solved])
        star = r"$^{\star}$" if len(solved) < 3 else ""
        return f"{len(solved)}/{len(cell)} ({med:.1f}){star}"
    return f"0/{len(cell)}"
rows_tex = []
for n in ns:
    for label, pool, tsel in (("60", bnd, lambda r: r["kstar_target"]),
                              ("600", b600, lambda r: str(2*int(r["tI"])))):
        cells = []
        for kt in kts:
            cell = [r for r in pool if int(r["nI"])+int(r["nJ"]) == n
                    and tsel(r) == str(kt)]
            solved = [r for r in cell if r["status"] == "OPTIMAL"
                      and r["timeout"] == "0"]
            cells.append(cell_tex(cell, solved))
        head = f"${n}$" if label == "60" else ""
        rows_tex.append(f"{head} & ${label}$ & " + " & ".join(cells) + r" \\")
    rows_tex.append(r"\addlinespace[2pt]")
rows_tex = rows_tex[:-1]
with open(os.path.join(OUT, "tab_q1_boundary.tex"), "w") as fh:
    fh.write(
r"""\begin{table}[H]
\centering
\caption{The boundary extension of Q1 as a function of the budget. Each
cell reports solved over attempted instances at the given $n=|I|+|J|$,
budget and covering-bound target $k^\ast$, with the median time in
seconds of the solved runs in parentheses. The $60$~s rows use the
three-seed refinement family and the $600$~s rows the ten-seed budget
family, both from the same generator. The tenfold budget displaces the
partially solved band from $n=24$ to $n=28$, where the solved count
again rises with the bound, and $n=40$ stays fully censored. Starred
cells have fewer than three solved runs and their medians are
individual observations.}
\label{tab:q1boundary}
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{llcccc}
\hline
$n$ & budget (s) & $k^\ast=4$ & $k^\ast=8$ & $k^\ast=12$ & $k^\ast=16$ \\
\hline
""" + "\n".join(rows_tex) + "\n" + r"""\hline
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Q2 codes
ctrl_mip = load("controlled_mip.csv")
ctrl_bd  = load("controlled_bd.csv")
main_xp  = [r for r in q1 if not (r["nK"] == "30" and r["nL"] == "3")]
xp_solved  = [r for r in main_xp if r["status"] == "OPTIMAL" and r["timeout"] == "0"]
mip_solved = [r for r in ctrl_mip if r["status"] == "OPTIMAL"]
bd_solved  = [r for r in ctrl_bd  if r["status"] == "OPTIMAL"]
def fmt_med(rs, key):
    if not rs: return "--"
    return round_half_up(median([float(r[key]) for r in rs]))
lines = [
    r"\textsc{Xp} & $60$ & $%d/%d$ & %s \\" % (len(xp_solved), len(main_xp),
        fmt_med(xp_solved, "time")),
    r"\textsc{Mip} & $600$ & $%d/%d$ & %s \\" % (len(mip_solved), len(ctrl_mip),
        fmt_med(mip_solved, "solver_time_s")),
    r"\textsc{Bd} & $600$ & $%d/%d$ & %s \\" % (len(bd_solved), len(ctrl_bd),
        fmt_med(bd_solved, "solver_time_s")),
]
with open(os.path.join(OUT, "tab_q2_codes.tex"), "w") as fh:
    fh.write(
r"""\begin{table}[H]
\centering
\caption{The three exact codes on the $400$-instance main grid of the
controlled family, whose graphs all have $n\ge 40$. Solved means proven
optimal within the per-code limit. The median time is over solved runs
only, in seconds. \textsc{Mip} closes every instance the enumeration
leaves censored, the empirical face of the regime border of Q1. The
three codes ran on different machines with different budgets, so the
table supports the solved-versus-censored contrast and no cross-code
time comparison.}
\label{tab:q2codes}
\begin{tabular}{lccc}
\hline
code & limit (s) & solved & median time (s) \\
\hline
""" + "\n".join(lines) + "\n" + r"""\hline
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------- Q3 groups
kstar = {r["instance"]: int(r["kstar_py"]) for r in load("mauri_kstar.csv")}
mauri = load("mauri_mip.csv")
def group_of(name):
    p = name.split("-")
    return f"{p[2]}-{p[3].split('.')[0]}"
groups = {}
for r in mauri:
    key = r["instance"] + ".txt" if r["instance"] + ".txt" in kstar else r["instance"]
    groups.setdefault(group_of(r["instance"]), []).append(
        (kstar[key], float(r["gap"]) * 100.0, float(r["solver_time_s"]),
         r["status"] == "OPTIMAL"))
order = ["50-5", "50-10", "100-5", "100-10"]
stats, pooled_k, pooled_g = [], [], []
for g in order:
    v = groups[g]
    ks = [a for a, _, _, _ in v]; gaps = [b for _, b, _, _ in v]
    nopt = sum(1 for *_, o in v if o)
    rho = spearman(ks, gaps); p = perm_p(ks, gaps)
    lo, hi = boot_ci(ks, gaps)
    pooled_k += ks; pooled_g += gaps
    stats.append([g, v, ks, gaps, nopt, rho, p, lo, hi])
pooled = spearman(pooled_k, pooled_g)
pooled_p = perm_p(pooled_k, pooled_g)
# Holm sobre os cinco testes de gap (quatro grupos + agrupado)
tests = [(st[0], st[6]) for st in stats] + [("pooled", pooled_p)]
srt = sorted(tests, key=lambda t: t[1]); m = len(srt); prev = 0.0; adj = {}
for i, (name, p) in enumerate(srt):
    a_ = min(1.0, max(prev, (m - i) * p)); prev = a_; adj[name] = a_
lines = []
for g, v, ks, gaps, nopt, rho, p, lo, hi in stats:
    lines.append(
        f"${g.split('-')[0]}$-${g.split('-')[1]}$ & ${len(v)}$ & "
        f"${min(ks)}$/${int(median(ks))}$/${max(ks)}$ & "
        f"${median(gaps):.1f}$ & ${nopt}$ & "
        f"${rho:+.2f}$ [${lo:+.2f}$, ${hi:+.2f}$] & "
        f"${adj[g]:.3f}$ " + chr(92)*2)
with open(os.path.join(OUT, "tab_q3_groups.tex"), "w") as fh:
    fh.write(
r"""\begin{table}[H]
\centering
\caption{The benchmark campaign by size group, $3{,}600$~s per instance.
Columns give the group $|I|$-$|L|$, the instance count, the minimum,
median and maximum of the covering bound $k^\ast$, the median terminal
gap in percent, the number of proven optima, the within-group Spearman
correlation between $k^\ast$ and the terminal gap, and its
Holm-adjusted two-sided permutation $p$-value. Correlations carry
percentile bootstrap $95\%%$ confidence intervals in brackets. Each
interval describes one group alone, and the within-group column is the
sharper instrument, the pooled value mixing the group effect with size.
The pooled correlation over all $100$ instances is $%.2f$,
Holm-adjusted $p=%.3f$.}
\label{tab:q3groups}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lcccccc}
\hline
group & $\#$ & $k^\ast$ min/med/max & gap (\%%) & opt.\ &
$\rho$ [$95\%%$ CI] & $p$ \\
\hline
""" % (pooled, adj["pooled"]) + "\n".join(lines) + "\n" + r"""\hline
\end{tabular}
\end{table}
""")

print("tabelas escritas em paper/tables/")
for f in sorted(os.listdir(OUT)):
    print(" ", f)
