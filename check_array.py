#!/usr/bin/env python3
"""Confronta solo l'array di temperature (ignora l'errore in riga 1 e il tempo in ultima riga).

Uso:
    python3 check_array.py <output_tuo> <output_atteso>
"""
import math
import sys

REL_TOL, ABS_TOL = 5e-5, 1e-9   # stesse tolleranze di test_smoke.py


def load(path):
    lines = open(path).read().splitlines()
    err = float(lines[0])
    rows = [tuple(map(float, l.split())) for l in lines[1:-1]]   # coppie (x, u)
    return err, rows


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    err_a, a = load(sys.argv[1])
    err_b, b = load(sys.argv[2])

    if len(a) != len(b):
        sys.exit(f"FAIL: numero di punti diverso: {len(a)} vs {len(b)}")

    bad = []
    for i, ((xa, ua), (xb, ub)) in enumerate(zip(a, b)):
        if not math.isclose(xa, xb, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            bad.append((i, "x", xa, xb))
        if not (math.isfinite(ua) and math.isclose(ua, ub, rel_tol=REL_TOL, abs_tol=ABS_TOL)):
            bad.append((i, "u", ua, ub))

    ok_err = math.isclose(err_a, err_b, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    print(f"errore: {err_a:.4e} vs {err_b:.4e}  -> {'OK' if ok_err else 'DIVERSO'}")

    if not bad:
        print(f"PASS: array di temperature identico ({len(a)} punti, bordi inclusi)")
        return

    print(f"FAIL: {len(bad)} valori diversi su {len(a)} punti. Primi 10:")
    for i, what, va, vb in bad[:10]:
        print(f"  punto {i:>7} ({what}): {va:.4e} vs {vb:.4e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
