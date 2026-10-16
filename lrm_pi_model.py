import csv
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter

SEED = 2026
BRIGADES = 6


def brigade_radii(j):
    return [2 * j + 2, 2 * j + 14, 2 * j + 26]


def pixel_counts(R, closed=True):
    inn = out = 0
    n = R + 1
    for i in range(-n, n):
        for j in range(-n, n):
            far = max(i * i, (i + 1) ** 2) + max(j * j, (j + 1) ** 2)
            nx = 0 if i <= 0 <= i + 1 else min(abs(i), abs(i + 1))
            ny = 0 if j <= 0 <= j + 1 else min(abs(j), abs(j + 1))
            near = nx * nx + ny * ny
            if far <= R * R:
                inn += 1
            if (near <= R * R) if closed else (near < R * R):
                out += 1
    return inn, out


def center_counts(R):
    c = 0
    for i in range(-R - 1, R + 1):
        for j in range(-R - 1, R + 1):
            if (i + 0.5) ** 2 + (j + 0.5) ** 2 <= R * R:
                c += 1
    return c


def lattice_points(R):
    return sum(2 * math.isqrt(R * R - a * a) + 1 for a in range(-R, R + 1))


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
        N = 4 * R * R
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
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.tif", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def figure_pixels(out, R=5):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    n = R + 1
    for i in range(-n, n):
        for j in range(-n, n):
            far = max(i * i, (i + 1) ** 2) + max(j * j, (j + 1) ** 2)
            nx = 0 if i <= 0 <= i + 1 else min(abs(i), abs(i + 1))
            ny = 0 if j <= 0 <= j + 1 else min(abs(j), abs(j + 1))
            near = nx * nx + ny * ny
            if far <= R * R:
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="0.45", edgecolor="0.3", lw=0.4))
            elif near < R * R:
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="0.82", edgecolor="0.3", lw=0.4))
            elif near == R * R:
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="white", edgecolor="0.3", lw=0.4, hatch="////"))
            else:
                ax.add_patch(Rectangle((i, j), 1, 1, facecolor="white", edgecolor="0.75", lw=0.3))
    t = [2 * math.pi * s / 400 for s in range(401)]
    ax.plot([R * math.cos(a) for a in t], [R * math.sin(a) for a in t], color="black", lw=1.4)
    ax.plot([0], [0], "ko", ms=3)
    ax.set_xlim(-n, n)
    ax.set_ylim(-n, n)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out, "Рисунок_1")


def figure_model(out):
    fig, ax = plt.subplots(figsize=(6.7, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 51)
    ax.axis("off")
    w, h = 29, 20.5
    xs = (0.8, 35.5, 70.2)
    top, bottom = 29.3, 1.0
    boxes = [
        (xs[0], top, "1. Формализация", "гипотезы $H_0$ и $H_1$,\nправило решения,\nручной подсчёт", "(формулирование)", "white", "-"),
        (xs[1], top, "2. Постановка", "технического\nзадания, протокол\nитераций", "(применение)", "white", "-"),
        (xs[2], top, "ИИ-ассистент", "пишет\nи исправляет код\nпо заданию", "(делегируется)", "0.85", "--"),
        (xs[2], bottom, "3. Верификация", "контрольные точки,\nтождество (3)", "(оценивание)", "white", "-"),
        (xs[1], bottom, "4. Интерпретация", "оценка π и δπ,\nзаключение по $H_0$", "(интерпретирование)", "white", "-"),
        (xs[0], bottom, "5. Рефлексия", "что сделал я,\nчто сделал ИИ,\nгде было трудно", "(метапознание)", "0.94", "-"),
    ]
    for x, y, title, body, tag, fc, ls in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                    facecolor=fc, edgecolor="black", lw=0.9, ls=ls))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=12)
        ax.text(x + w / 2, y + h / 2 + 0.2, body, ha="center", va="center", fontsize=12, linespacing=1.05)
        ax.text(x + w / 2, y + 1.2, tag, ha="center", va="bottom", fontsize=12, color="0.3")

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                     mutation_scale=12, lw=0.9, color="black"))

    mid_top, mid_bottom = top + h / 2, bottom + h / 2
    arrow((xs[0] + w + 0.6, mid_top), (xs[1] - 0.6, mid_top))
    arrow((xs[1] + w + 0.6, mid_top), (xs[2] - 0.6, mid_top))
    arrow((xs[2] + w / 2, top - 0.6), (xs[2] + w / 2, bottom + h + 0.6))
    arrow((xs[2] - 0.6, mid_bottom), (xs[1] + w + 0.6, mid_bottom))
    arrow((xs[1] - 0.6, mid_bottom), (xs[0] + w + 0.6, mid_bottom))
    arrow((xs[2] + 3, bottom + h + 0.6), (xs[1] + w - 3, top - 0.6), rad=-0.2)
    ax.text(xs[1] + 2, (top + bottom + h) / 2, "расхождение, новая итерация", ha="center", va="center", fontsize=12)
    save(fig, out, "Схема_компонентов")


def figure_results(out, rows, k, cum_pix, cum_mc):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.7, 2.8), gridspec_kw={"wspace": 0.42})
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
    ax.text(0.5, -0.33, "а", transform=ax.transAxes, ha="center", fontsize=12)
    xs = list(range(1, BRIGADES + 1))
    bx.axhline(math.pi, color="black", lw=0.9)
    bx.axhline(3.0, color="black", lw=0.9, ls="--")
    bx.text(6.45, math.pi + 0.01, "π", fontsize=12, va="bottom")
    bx.text(6.05, 3.0 + 0.01, "$k$ = 3", fontsize=12, va="bottom")
    bx.errorbar([x - 0.1 for x in xs], [c[0] for c in cum_pix], yerr=[2 * c[1] for c in cum_pix], fmt="s", ms=4,
                color="black", mfc="black", elinewidth=0.9, capsize=2.5, label="пиксели")
    bx.errorbar([x + 0.1 for x in xs], [c[0] for c in cum_mc], yerr=[2 * c[1] for c in cum_mc], fmt="o", ms=4,
                color="0.35", mfc="white", elinewidth=0.9, capsize=2.5, label="Монте-Карло")
    bx.set_xlim(0.5, 7.2)
    bx.set_ylim(2.80, 3.45)
    bx.set_xticks(xs)
    bx.set_xlabel("число бригад")
    bx.set_ylabel("$k$")
    bx.yaxis.set_major_formatter(FuncFormatter(lambda v, p: comma(v, 1)))
    bx.legend(fontsize=12, frameon=False, loc="lower right", handletextpad=0.3, borderaxespad=0.2)
    bx.text(0.5, -0.33, "б", transform=bx.transAxes, ha="center", fontsize=12)
    save(fig, out, "Рисунок_2")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    rows = []
    for j in range(1, BRIGADES + 1):
        for R in brigade_radii(j):
            inn, o = pixel_counts(R, True)
            _, o_open = pixel_counts(R, False)
            S = (inn + o) / 2
            rows.append({
                "brigade": j, "R": R, "S_inn": inn, "S_out": o, "S_out_open": o_open,
                "S_mid": S, "dS": (o - inn) / 2, "dev": S - math.pi * R * R,
                "S_centers": center_counts(R), "N_lattice": lattice_points(R),
            })

    with open(out / "данные_модельного_эксперимента.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["бригада", "R", "S_inn", "S_out", "<S>", "dS", "<S>-piR2", "S_out (открытое соглашение)", "S по центрам пикселей", "N(R) точек решётки"])
        for r in sorted(rows, key=lambda r: r["R"]):
            w.writerow([r["brigade"], r["R"], r["S_inn"], r["S_out"], comma(r["S_mid"], 1), comma(r["dS"], 1),
                        comma(r["dev"], 2), r["S_out_open"], r["S_centers"], r["N_lattice"]])

    print("identity S_out-S_inn=8R+4:", all(r["S_out"] - r["S_inn"] == 8 * r["R"] + 4 for r in rows))
    print("identity <S>=N(R)+1:", all(r["S_mid"] == r["N_lattice"] + 1 for r in rows))
    print("min ratio dS/|dev|:", round(min(r["dS"] / abs(r["dev"]) for r in rows), 2))
    print("dev range:", round(min(r["dev"] for r in rows), 2), round(max(r["dev"] for r in rows), 2))

    mc_x, mc_y, mc_d = monte_carlo([R for j in range(1, BRIGADES + 1) for R in brigade_radii(j)], SEED)
    cum_pix, cum_mc = [], []
    print("brigades circles k dk dk_max |k-pi| reject3")
    for j in range(1, BRIGADES + 1):
        sel = [r for r in rows if r["brigade"] <= j]
        k, dk, dkm = fit([r["R"] ** 2 for r in sel], [r["S_mid"] for r in sel], [r["dS"] for r in sel])
        cum_pix.append((k, dk))
        m = 3 * j
        km, dkm_mc, _ = fit(mc_x[:m], mc_y[:m], mc_d[:m])
        cum_mc.append((km, dkm_mc))
        print(j, 3 * j, round(k, 4), round(dk, 4), round(dkm, 4), round(abs(k - math.pi), 4), abs(k - 3) > 2 * dk,
              "| MC", round(km, 4), round(dkm_mc, 4))
    k_all, dk_all, _ = fit([r["R"] ** 2 for r in rows], [r["S_mid"] for r in rows], [r["dS"] for r in rows])
    for name, v in (("22/7", 22 / 7), ("(16/9)^2", (16 / 9) ** 2), ("25/8", 25 / 8)):
        print(name, "rejected:", abs(k_all - v) > 2 * dk_all)
    ko, dko, _ = fit([r["R"] ** 2 for r in rows], [(r["S_inn"] + r["S_out_open"]) / 2 for r in rows],
                     [(r["S_out_open"] - r["S_inn"]) / 2 for r in rows])
    print("open convention:", round(ko, 4), round(dko, 4))
    sxx = sum(r["R"] ** 4 for r in rows)
    print("centers convention:", round(sum(r["R"] ** 2 * r["S_centers"] for r in rows) / sxx, 4))
    runs = [fit(*monte_carlo([R for j in range(1, BRIGADES + 1) for R in brigade_radii(j)], s))[0] for s in range(1, 201)]
    mean = sum(runs) / len(runs)
    sd = math.sqrt(sum((v - mean) ** 2 for v in runs) / (len(runs) - 1))
    print("MC 200 seeds: mean", round(mean, 4), "sd", round(sd, 4), "min", round(min(runs), 4), "max", round(max(runs), 4))
    worst = max(abs(lattice_points(R) + 1 - math.pi * R * R) / math.sqrt(R) for R in range(1, 1501))
    negative = sum(1 for R in range(1, 1501) if lattice_points(R) + 1 < math.pi * R * R)
    print("R<=1500: max|dev|/sqrt(R)", round(worst, 2), "negative", negative, "of 1500")

    figure_pixels(out)
    figure_model(out)
    figure_results(out, rows, k_all, cum_pix, cum_mc)


if __name__ == "__main__":
    main()
