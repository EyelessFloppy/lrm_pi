import csv
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator

SEED = 2026
BRIGADES = 6
PER_BRIGADE = 3
R_MIN, R_MAX = 4.0, 38.0
Q = 10
CONTROL = (45, 65)


def draw_radii(seed=SEED):
    rng = random.Random(seed)
    return [round(rng.uniform(R_MIN, R_MAX) * Q) for _ in range(BRIGADES * PER_BRIGADE)]


def fl(m, k):
    v = m * m - Q * Q * k * k
    return math.isqrt(v) // Q if v >= 0 else -1


def exact_quarter(m):
    r = m // Q
    inn = sum(fl(m, k) for k in range(1, r + 1))
    out = sum(fl(m, k) + 1 for k in range(0, r + 1))
    return inn, out


def classify(m, i, j):
    far = Q * Q * ((i + 1) ** 2 + (j + 1) ** 2)
    near = Q * Q * (i * i + j * j)
    mm = m * m
    if far <= mm:
        return "inner"
    if near < mm:
        return "boundary"
    if near == mm:
        return "touch"
    return "outside"


def program_quarter(m, variant="correct"):
    n = m // Q + 2
    inn = out = 0
    cols = range(m // Q) if variant == "offbyone" else range(n)
    for i in range(n):
        for j in range(n):
            if classify(m, i, j) == "inner":
                inn += 1
    for i in cols:
        for j in range(n):
            c = classify(m, i, j)
            if c in ("inner", "boundary") or (c == "touch" and variant != "open"):
                out += 1
    return inn, out


def centers_full(m):
    n = m // Q + 1
    c = 0
    for i in range(-n, n):
        for j in range(-n, n):
            if Q * Q * ((2 * i + 1) ** 2 + (2 * j + 1) ** 2) <= 4 * m * m:
                c += 1
    return c


def fit(xs, ys, ds):
    sxx = sum(x * x for x in xs)
    k = sum(x * y for x, y in zip(xs, ys)) / sxx
    dk = math.sqrt(sum((x * d) ** 2 for x, d in zip(xs, ds))) / sxx
    dk_max = sum(x * d for x, d in zip(xs, ds)) / sxx
    return k, dk, dk_max


def monte_carlo(radii, seed):
    rng = random.Random(seed)
    xs, ys, ds = [], [], []
    for R in radii:
        N = round(4 * R * R)
        hits = 0
        for _ in range(N):
            x = rng.uniform(-R, R)
            y = rng.uniform(-R, R)
            if x * x + y * y <= R * R:
                hits += 1
        p = hits / N
        xs.append(R * R)
        ys.append(4 * R * R * p)
        ds.append(4 * R * R * math.sqrt(p * (1 - p) / N))
    return xs, ys, ds


def comma(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def setup_fonts():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "axes.linewidth": 0.8,
        "mathtext.fallback": "stix",
    })


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.tif", dpi=600, bbox_inches="tight", facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def draw_quarter(ax, m, n_cells):
    R = m / Q
    for i in range(n_cells):
        for j in range(n_cells):
            c = classify(m, i, j)
            if c == "inner":
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="0.45", edgecolor="0.3", lw=0.4))
            elif c == "boundary":
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="0.82", edgecolor="0.3", lw=0.4))
            elif c == "touch":
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="white", edgecolor="0.3", lw=0.4, hatch="////"))
            else:
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="white", edgecolor="0.75", lw=0.3))
    t = [math.pi / 2 * s / 200 for s in range(201)]
    ax.plot([R * math.cos(a) for a in t], [R * math.sin(a) for a in t], color="black", lw=1.4)
    ax.plot([0], [0], "ko", ms=2.5)
    ax.set_xlim(-0.15, n_cells + 0.15)
    ax.set_ylim(-0.15, n_cells + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def figure_pixels(out, panels):
    fig, axes = plt.subplots(1, len(panels), figsize=(5.8, 2.15), gridspec_kw={"wspace": 0.12})
    n_cells = max(m // Q + 2 for m, _ in panels)
    for ax, (m, letter) in zip(axes, panels):
        draw_quarter(ax, m, n_cells)
        ax.text(0.5, -0.1, f"{letter}", transform=ax.transAxes, ha="center", va="top", fontsize=12)
    save(fig, out, "Рисунок_1")


def figure_results(out, rows, k, cum_pix, cum_mc):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.0, 2.15), gridspec_kw={"wspace": 0.42})
    ax.errorbar([r["R"] ** 2 for r in rows], [r["S_mid"] for r in rows], yerr=[r["dS"] for r in rows],
                fmt="o", ms=4, mfc="white", mec="black", ecolor="0.4", elinewidth=0.7, capsize=2)
    xmax = 1500
    ax.plot([0, xmax], [0, k * xmax], color="black", lw=1.0)
    ax.text(80, 4300, f"$y$ = {comma(k, 3)}$x$", fontsize=12)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 5000)
    ax.set_xticks([0, 500, 1000, 1500])
    ax.set_xlabel("$R$²")
    ax.set_ylabel(r"$\langle S\rangle$")
    ax.text(0.5, -0.45, "а", transform=ax.transAxes, ha="center", fontsize=12)
    xs = list(range(1, BRIGADES + 1))
    bx.axhline(math.pi, color="black", lw=0.9)
    bx.axhline(3.0, color="black", lw=0.9, ls="--")
    bx.text(6.45, math.pi + 0.01, "π", fontsize=12, va="bottom")
    bx.text(6.05, 3.0 + 0.01, "$k$ = 3", fontsize=12, va="bottom")
    bx.errorbar([x - 0.1 for x in xs], [c[0] for c in cum_pix], yerr=[2 * c[1] for c in cum_pix], fmt="s", ms=4,
                color="black", mfc="black", elinewidth=0.9, capsize=2.5, label="пиксели")
    bx.errorbar([x + 0.1 for x in xs], [c[0] for c in cum_mc], yerr=[2 * c[1] for c in cum_mc], fmt="o", ms=4,
                color="0.35", mfc="white", elinewidth=0.9, capsize=2.5, label="Монте-Карло")
    lo = min(min(c[0] - 2 * c[1] for c in cum_pix), min(c[0] - 2 * c[1] for c in cum_mc), 3.0)
    hi = max(max(c[0] + 2 * c[1] for c in cum_pix), max(c[0] + 2 * c[1] for c in cum_mc))
    bx.set_xlim(0.5, 7.2)
    bx.set_ylim(math.floor((lo - 0.06) * 10) / 10 - 0.2, math.ceil((hi + 0.04) * 10) / 10)
    bx.set_xticks(xs)
    bx.set_xlabel("число бригад")
    bx.set_ylabel("$k$")
    bx.yaxis.set_major_locator(MultipleLocator(0.2))
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, p: comma(v, 1)))
    bx.legend(fontsize=12, frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    bx.text(0.5, -0.45, "б", transform=bx.transAxes, ha="center", fontsize=12)
    save(fig, out, "Рисунок_2")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    for m in CONTROL + (50,):
        inn, o = exact_quarter(m)
        print("control R =", m / Q, "quarter Sinn", inn, "Sout", o, "full", 4 * inn, 4 * o,
              "<S>", 2 * (inn + o), "dS", 2 * (o - inn), "piR2", round(math.pi * (m / Q) ** 2, 2))

    ms = draw_radii()
    rows = []
    for idx, m in enumerate(ms):
        R = m / Q
        qi, qo = exact_quarter(m)
        pi_, po = program_quarter(m)
        oi, oo = program_quarter(m, "open")
        bi, bo = program_quarter(m, "offbyone")
        inn, o = 4 * qi, 4 * qo
        rows.append({
            "brigade": idx // PER_BRIGADE + 1, "m": m, "R": R, "S_inn": inn, "S_out": o,
            "S_mid": (inn + o) / 2, "dS": (o - inn) / 2, "dev": (inn + o) / 2 - math.pi * R * R,
            "prog_ok": (pi_, po) == (qi, qo), "open_ok": (oi, oo) == (qi, qo), "obo_ok": (bi, bo) == (qi, qo),
            "S_out_obo": 4 * bo, "S_centers": centers_full(m),
        })

    with open(out / "данные_модельного_эксперимента.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["бригада", "R", "S_inn", "S_out", "<S>", "dS", "<S>-piR2", "S_out (ошибка в цикле)",
                    "S по центрам пикселей"])
        for r in rows:
            w.writerow([r["brigade"], comma(r["R"], 1), r["S_inn"], r["S_out"], comma(r["S_mid"], 1),
                        comma(r["dS"], 1), comma(r["dev"], 2), r["S_out_obo"], r["S_centers"]])

    print("radii by brigade:", [[comma(r["R"], 1) for r in rows if r["brigade"] == j] for j in range(1, BRIGADES + 1)])
    print("sorted radii:", sorted(r["R"] for r in rows))
    print("integer radii among them:", [r["R"] for r in rows if r["m"] % Q == 0])
    print("correct program == exact sums:", all(r["prog_ok"] for r in rows))
    print("open program == exact sums on random radii:", all(r["open_ok"] for r in rows))
    print("off-by-one program == exact sums:", sum(r["obo_ok"] for r in rows), "of", len(rows))
    print("identity S_out-S_inn = 8[R]+4:", all(r["S_out"] - r["S_inn"] == 8 * (r["m"] // Q) + 4 for r in rows))
    print("dev range:", round(min(r["dev"] for r in rows), 2), round(max(r["dev"] for r in rows), 2))
    print("dS range:", min(r["dS"] for r in rows), max(r["dS"] for r in rows))
    print("max |dev|/dS:", round(max(abs(r["dev"]) / r["dS"] for r in rows), 3))

    for v in ("correct", "open", "offbyone"):
        print(v, "R=5 quarter:", program_quarter(50, v), "exact", exact_quarter(50))
    print("open R=sqrt5 test needs node: R^2=5 nodes (1,2),(2,1)")

    radii = [r["R"] for r in rows]
    mc_x, mc_y, mc_d = monte_carlo(radii, SEED)
    cum_pix, cum_mc = [], []
    print("brigades circles k dk dk_max |k-pi| reject3 | MC k dk")
    for j in range(1, BRIGADES + 1):
        sel = [r for r in rows if r["brigade"] <= j]
        k, dk, dkm = fit([r["R"] ** 2 for r in sel], [r["S_mid"] for r in sel], [r["dS"] for r in sel])
        cum_pix.append((k, dk))
        m = PER_BRIGADE * j
        km, dkm_mc, _ = fit(mc_x[:m], mc_y[:m], mc_d[:m])
        cum_mc.append((km, dkm_mc))
        print(j, m, round(k, 4), round(dk, 4), round(dkm, 4), round(abs(k - math.pi), 4), abs(k - 3) > 2 * dk,
              "| MC", round(km, 4), round(dkm_mc, 4))
    xs = [r["R"] ** 2 for r in rows]
    k_all, dk_all, dkm_all = fit(xs, [r["S_mid"] for r in rows], [r["dS"] for r in rows])
    print("all: k", round(k_all, 4), "dk", round(dk_all, 4), "dk_max", round(dkm_all, 4), "2dk", round(2 * dk_all, 4))
    for name, v in (("3", 3.0), ("22/7", 22 / 7), ("(16/9)^2", (16 / 9) ** 2), ("25/8", 25 / 8), ("3.16", 3.16)):
        print(name, "rejected:", abs(k_all - v) > 2 * dk_all, "dist", round(abs(k_all - v), 4))
    sxx = sum(x * x for x in xs)
    k_c = sum(x * r["S_centers"] for x, r in zip(xs, rows)) / sxx
    print("centers convention k:", round(k_c, 4))
    ko, dko, _ = fit(xs, [(r["S_inn"] + r["S_out_obo"]) / 2 for r in rows],
                     [(r["S_out_obo"] - r["S_inn"]) / 2 for r in rows])
    print("off-by-one k:", round(ko, 4), round(dko, 4))
    km_all, dkm_mc_all, _ = fit(mc_x, mc_y, mc_d)
    print("MC seed", SEED, ":", round(km_all, 4), round(dkm_mc_all, 4), "ratio dk/dk_mc", round(dk_all / dkm_mc_all, 2))
    runs = [fit(*monte_carlo(radii, s))[0] for s in range(1, 201)]
    mean = sum(runs) / len(runs)
    sd = math.sqrt(sum((v - mean) ** 2 for v in runs) / (len(runs) - 1))
    print("MC 200 seeds: mean", round(mean, 4), "sd", round(sd, 4), "min", round(min(runs), 4), "max", round(max(runs), 4))

    rng = random.Random(1)
    big = [round(rng.uniform(R_MIN, R_MAX) * Q) for _ in range(5000)]
    worst = 0.0
    for m in big:
        qi, qo = exact_quarter(m)
        dev = 2 * (qi + qo) - math.pi * (m / Q) ** 2
        worst = max(worst, abs(dev) / (2 * (qo - qi)))
    print("5000 random radii: max |dev|/dS", round(worst, 3))

    for R in (100, 1000):
        m = Q * R
        res = {}
        for v in ("correct", "open", "offbyone"):
            si, so = program_quarter(m, v) if R == 100 else exact_quarter(m)
            res[v] = round(2 * (si + so) / R ** 2, 5)
        print("R =", R, res)

    figure_pixels(out, [(CONTROL[0], "а"), (CONTROL[1], "б"), (50, "в")])
    figure_results(out, rows, k_all, cum_pix, cum_mc)


if __name__ == "__main__":
    main()
