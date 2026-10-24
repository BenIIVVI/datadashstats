"""The model-side slide charts, in plain matplotlib. From the repo folder, after
python r/slide_figure_data.py:

    python r/slide_figures_matplotlib.py     writes figures/slides/{split,boosting,steps,recency,onetree,importance}.png

The other slide charts come from r/slide_figures.R.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "figure_data"
OUT = ROOT / "figures" / "slides"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=400)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # how a tree picks its first question: error removed by every possible cut
    sp = pd.read_csv(DATA / "split.csv")
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    for name, rows in sp.groupby("feature", sort=False):
        best = rows.loc[rows["drop"].idxmax()]
        label = f"{name} (best cut {best.cut:,.0f})" if name == "Size" else f"{name} (best cut ${best.cut / 1e3:,.0f}k)"
        line, = ax.plot(100 * rows["q"], rows["drop"], label=label)
        ax.plot(100 * best.q, best["drop"], "o", color=line.get_color())
    ax.set_xlabel("where the cut falls (% of sales below it)")
    ax.set_ylabel("% of error removed")
    ax.legend(fontsize=8)
    save(fig, "split")

    # typical miss as trees are added
    bo = pd.read_csv(DATA / "boosting.csv")
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(bo["trees"], bo["mdape"], marker="o")
    ax.set_xscale("log")
    ax.set_xlim(0.7, 4000)
    ax.set_ylim(0, 40)
    ax.set_xlabel("number of trees")
    ax.set_ylabel("MdAPE (%)")
    for n in (1, 100, 2000):
        v = bo.loc[bo["trees"] == n, "mdape"].iloc[0]
        ax.annotate(f"{v:.1f}%", (n, v), textcoords="offset points", xytext=(4, 6), fontsize=8)
    save(fig, "boosting")

    # the Elmbank house's value as trees are added
    st = pd.read_csv(DATA / "steps.csv")
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(st["trees"] + 1, st["value"] / 1e3, marker="o")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 11, 101, 1001, 2001], ["0", "1", "10", "100", "1000", "2000"])
    ax.set_xlabel("trees added so far")
    ax.set_ylabel("value ($ thousands)")
    for n, txt in ((0, "start"), (50, "50 trees"), (2000, "2000 trees")):
        v = st.loc[st["trees"] == n, "value"].iloc[0] / 1e3
        ax.annotate(f"{txt}: ${v:,.0f}k", (n + 1, v), textcoords="offset points", xytext=(4, 6), fontsize=8)
    save(fig, "steps")

    # recency weights
    m = np.arange(0, 65)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    ax.plot(m, 0.5 ** (m / 12))
    ax.plot([0, 12, 24, 36], [1, 0.5, 0.25, 0.125], "o", color="C0")
    ax.plot([64], [0.5 ** (64 / 12)], "o", color="C1")
    ax.annotate("Jan 2021: 0.025", (64, 0.5 ** (64 / 12)), textcoords="offset points", xytext=(-80, 10), fontsize=8)
    ax.set_xlabel("months before the data ends")
    ax.set_ylabel("weight")
    save(fig, "recency")

    # one big tree vs boosting
    ot = pd.read_csv(DATA / "onetree.csv")
    fig, ax = plt.subplots(figsize=(8.32, 2.0))
    bars = ax.barh(ot["model"][::-1], ot["mdape"][::-1])
    ax.bar_label(bars, fmt="%.2f%%", fontsize=8, padding=3)
    ax.set_xlim(0, 13)
    ax.set_xlabel("MdAPE (%), 2024 summer backtest")
    save(fig, "onetree")

    # permutation importance by feature family
    im = pd.read_csv(DATA / "importance.csv").sort_values("added")
    fig, ax = plt.subplots(figsize=(8.32, 2.1))
    bars = ax.barh(im["family"], im["added"])
    ax.bar_label(bars, fmt="+%.1f", fontsize=8, padding=3)
    ax.set_xlim(0, 18.5)
    ax.set_xlabel("increase in MdAPE when the family is shuffled (percentage points)")
    save(fig, "importance")
    print(f"wrote 6 charts to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
