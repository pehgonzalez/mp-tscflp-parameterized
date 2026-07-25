"""Extracts the SP-TSCFLP instances of Fernandes et al. (2014) from the MP benchmark.

Mauri et al. (2021, sec. 6.1): product 1 of each MP instance keeps the original
single-product data. Writing the L=1 slice therefore reconstructs the original set.
Usage: python extract_sp.py <data_dir> <out_dir>   (uses only *-5.txt files)
"""
import pathlib
import sys


def extract(src: pathlib.Path, dst: pathlib.Path) -> None:
    tok = src.read_text().split()
    it = iter(tok)
    n = lambda: int(next(it))
    I, J, K, L = n(), n(), n(), n()
    q = [[next(it) for _ in range(L)] for _ in range(K)]
    b, f = [], []
    for _ in range(I):
        b.append([next(it) for _ in range(L)])
        f.append(next(it))
    c = [[[next(it) for _ in range(J)] for _ in range(I)] for _ in range(L)]
    p, g = [], []
    for _ in range(J):
        p.append([next(it) for _ in range(L)])
        g.append(next(it))
    d = [[[next(it) for _ in range(K)] for _ in range(J)] for _ in range(L)]

    out = [f"{I} {J} {K} 1"]
    out += [q[k][0] for k in range(K)]
    out += [f"{b[i][0]} {f[i]}" for i in range(I)]
    out += [" ".join(c[0][i]) for i in range(I)]
    out += [f"{p[j][0]} {g[j]}" for j in range(J)]
    out += [" ".join(d[0][j]) for j in range(J)]
    dst.write_text("\n".join(out) + "\n")


def main():
    data, outdir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(data.glob("PSC*-5.txt"))  # L=5 files carry the original data
    for src in files:
        name = src.name.replace("-5.txt", "-SP.txt")
        extract(src, outdir / name)
        print(name)
    print(f"{len(files)} SP instances written to {outdir}")


if __name__ == "__main__":
    main()
