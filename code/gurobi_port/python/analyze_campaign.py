"""Análise da campanha (esquema pós-auditoria de results.csv).

Uso:  python analyze_campaign.py [results.csv] [datetime_min]
- Usa total_wall_s (orçamento completo, inclui fase lagrangiana) nas comparações.
- Pareia (instância, seed): fechados comparados por tempo; abertos por gap.
- ABORTA se os pares misturarem time_limit/threads/gurobi_version diferentes.
- verified: OPTIMAL exige |verified-obj| <= 0.5; FEASIBLE aceita verified <= obj + 0.5
  (folga de θ/CNUF é reportada, não é erro).
Saída: campaign_summary.md ao lado do CSV.
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

NUM = ("obj", "bound", "gap", "solver_time_s", "total_wall_s", "lag_lb", "lag_ub",
       "lag_time_s", "heuristic_cost", "verified_cost", "time_limit_s")


def load(path, dt_min):
    rows = []
    with open(path, newline="") as f:
        for ln, r in enumerate(csv.DictReader(f), 2):
            try:
                if dt_min and r["datetime"] < dt_min:
                    continue
                for k in NUM:
                    r[k] = float(r[k])
                r["method"] = int(r["method"])
                r["seed"] = int(r["seed"])
                rows.append(r)
            except (KeyError, ValueError) as e:
                print(f"AVISO: linha {ln} ignorada ({e})")
    return rows


def verify_state(r):
    if r["verified_cost"] < 0:
        return "SEM_SOLUCAO"
    d = r["obj"] - r["verified_cost"]
    if r["status"] == "OPTIMAL":
        return "OK" if abs(d) <= 0.5 else "ERRO_DIVERGENCIA"
    return "OK" if d >= -0.5 else "ERRO_DIVERGENCIA"  # verified <= obj + 0.5


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "logs/results.csv"
    dt_min = sys.argv[2] if len(sys.argv) > 2 else ""
    rows = [r for r in load(path, dt_min)
            if r["mode"] == "exact" and r["method"] in (1, 2)]
    if not rows:
        print("Nenhuma linha elegível.")
        return

    # Consistência de configuração (audit #5): um único valor por campo, ou aborta.
    for field in ("time_limit_s", "threads", "gurobi_version"):
        vals = {r[field] for r in rows}
        if len(vals) > 1:
            sys.exit(f"ERRO: campo {field} mistura valores {vals}. "
                     f"Filtre por datetime_min ou separe os CSVs.")

    best = {}
    for r in rows:  # cronológico: última execução por célula vence
        best[(r["instance"], r["method"], r["seed"])] = r

    insts = sorted({k[0] for k in best})
    seeds = sorted({k[2] for k in best})
    out = [f"# Campanha ({Path(path).name}, desde '{dt_min or 'início'}') — "
           "m1 (Benders) vs m2 (Lagrangian-guided)", ""]
    out.append("| instância | seed | m | status | obj | bound | wall(s) | cortes | verif |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    agg = defaultdict(list)
    errors = []
    for inst in insts:
        for s in seeds:
            for m in (2, 1):
                r = best.get((inst, m, s))
                if not r:
                    continue
                vs = verify_state(r)
                if vs == "ERRO_DIVERGENCIA":
                    errors.append((inst, s, m))
                out.append(f"| {inst} | {s} | {m} | {r['status']} | {r['obj']:.0f} | "
                           f"{r['bound']:.2f} | {r['total_wall_s']:.1f} | "
                           f"{r['benders_cuts']} | {vs} |")
                agg[m].append(r)
    out.append("")
    for m in (2, 1):
        rs = agg[m]
        if not rs:
            continue
        closed = [r for r in rs if r["status"] == "OPTIMAL"]
        open_ = [r for r in rs if r["status"] != "OPTIMAL"]
        line = f"**m{m}**: {len(closed)}/{len(rs)} células fechadas"
        if closed:
            line += f"; wall mediano p/ fechar {statistics.median(r['total_wall_s'] for r in closed):.0f}s"
        if open_:
            line += f"; gap mediano das abertas {statistics.median(r['gap'] for r in open_):.4%}"
        out.append(line + ".")

    # Pareado por (instância, seed): fechados por tempo, abertos por gap (audit #20).
    pairs = [(best[(i, 2, s)], best[(i, 1, s)]) for i in insts for s in seeds
             if (i, 2, s) in best and (i, 1, s) in best]
    if pairs:
        w2 = w1 = tie = 0
        speedups, diverg = [], []
        for a, b in pairs:  # a = m2, b = m1
            ca, cb = a["status"] == "OPTIMAL", b["status"] == "OPTIMAL"
            if ca and cb:
                speedups.append(b["total_wall_s"] / max(a["total_wall_s"], 0.1))
                if abs(a["obj"] - b["obj"]) > 0.5:
                    diverg.append((a["instance"], a["seed"]))
                d = b["total_wall_s"] - a["total_wall_s"]
                w2, w1, tie = w2 + (d > 1), w1 + (d < -1), tie + (abs(d) <= 1)
            elif ca != cb:
                w2, w1 = w2 + ca, w1 + cb
            else:
                d = b["gap"] - a["gap"]
                eps = 1e-6
                w2, w1, tie = w2 + (d > eps), w1 + (d < -eps), tie + (abs(d) <= eps)
        out.append(f"\nPareado ({len(pairs)} pares instância×seed): "
                   f"m2 {w2} × {w1} m1 ({tie} empates).")
        if speedups:
            out.append(f"Speedup mediano m2 sobre m1 (ambos fechados, wall total): "
                       f"{statistics.median(speedups):.2f}x.")
        out.append("CONSISTÊNCIA: " + ("ótimos idênticos em todos os pares fechados."
                   if not diverg else f"DIVERGÊNCIA REAL em {diverg} — INVESTIGAR!"))
    if errors:
        out.append(f"\n**ERROS DE VERIFICAÇÃO** (investigar antes de usar): {errors}")
    text = "\n".join(out)
    dst = Path(path).parent / "campaign_summary.md"
    dst.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
