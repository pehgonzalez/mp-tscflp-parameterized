#!/usr/bin/env python3
"""Join the paired records of the covering-pruning ablation.

Reads the one-line records written by code/run_p1_ablation.sh into
results/p1_ablation_lines/, one .on file (full algorithm) and one .off file
(MPTSCFL_NO_P1=1) per instance, writes the joined table to
results/p1_ablation.csv and prints the summary quoted in Section 6.2 of the
paper. No solver run is performed here.
"""
import csv
import os
import statistics as st

LINES = os.path.join(os.path.dirname(__file__), '..', 'results', 'p1_ablation_lines')
OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'p1_ablation.csv')

FIELDS = ['status', 'obj', 'kstar', 'nodes', 'time', 'p1', 'p2i', 'p2b', 'p3']


def parse(path):
    text = open(path).read().strip()
    if not text:
        return None
    return dict(kv.split('=', 1) for kv in text.split())


def load():
    pairs = []
    for name in sorted(os.listdir(LINES)):
        if not name.endswith('.on'):
            continue
        base = name[:-3]
        on = parse(os.path.join(LINES, name))
        off_path = os.path.join(LINES, base + '.off')
        off = parse(off_path) if os.path.exists(off_path) else None
        if on is None or off is None:
            print('incomplete pair skipped: %s' % base)
            continue
        pairs.append((base, on, off))
    return pairs


def write_csv(pairs):
    header = ['instance', 'kstar']
    for f in FIELDS:
        if f == 'kstar':
            continue
        header += [f + '_on', f + '_off']
    with open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for base, on, off in pairs:
            row = [base, on['kstar']]
            for f in FIELDS:
                if f == 'kstar':
                    continue
                row += [on[f], off[f]]
            w.writerow(row)


def summarize(pairs):
    term = [p for p in pairs if p[1]['status'] == 'OPTIMAL' and p[2]['status'] == 'OPTIMAL']
    cens = [p for p in pairs if p not in term]
    print('pairs %d, both terminate %d, at least one censored %d'
          % (len(pairs), len(term), len(cens)))
    print('terminating subset')
    for field in ('nodes', 'obj', 'p2b', 'p3'):
        agree = sum(1 for _, o, f in term if o[field] == f[field])
        print('  identical %-6s %d/%d' % (field, agree, len(term)))
    migr = sum(1 for _, o, f in term
               if int(f['p2i']) - int(o['p2i']) == int(o['p1']))
    print('  exact p1 -> p2i migration %d/%d' % (migr, len(term)))
    ton = sum(float(o['time']) for _, o, _ in term)
    toff = sum(float(f['time']) for _, _, f in term)
    slower = sum(1 for _, o, f in term if float(f['time']) > float(o['time']))
    print('  total time %.1f s with the rule, %.1f s without it (ratio %.4f)'
          % (ton, toff, toff / ton))
    print('  slower without the rule in %d of %d runs' % (slower, len(term)))
    print('share of nodes decided by the covering rule')
    by_k = {}
    for _, on, _ in pairs:
        by_k.setdefault(int(on['kstar']), []).append(int(on['p1']) / int(on['nodes']))
    for k in sorted(by_k):
        print('  k* = %-3d median share %.4f' % (k, st.median(by_k[k])))


if __name__ == '__main__':
    pairs = load()
    write_csv(pairs)
    summarize(pairs)
    print('wrote %s' % os.path.normpath(OUT))
