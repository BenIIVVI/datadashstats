"""Numbers behind the off-market slide charts, for r/slide_figures.R. From the repo folder:

    python r/slide_figure_data.py      writes outputs/figure_data/*.csv (about 15 minutes)
    Rscript r/slide_figures.R          draws figures/slides/*.png from them

Every model here is trained only on sales before June 1 of its season. The worked example is
one real home from the 2024 summer backtest: SYN-0129354, a detached house in Elmbank.
Also needs outputs/backtest_predictions.csv from `python run.py backtest`.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import data      # noqa: E402
import models    # noqa: E402
from features import MarketIndex   # noqa: E402

OUT = ROOT / "outputs" / "figure_data"
EXAMPLE = "SYN-0129354"
TYPE_NAMES = {"Co-Op Apt": "Co-op", "Detached": "Detached", "Semi-Detached": "Semi-detached",
              "Freehold Townhouse": "Freehold townhouse", "Condo Townhouse": "Condo townhouse",
              "Condo Apt": "Condo apartment"}


def mdape(predicted, sold):
    return round(float(100 * np.median(np.abs(predicted / sold - 1))), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    listings = data.load("listings")
    market_index(listings)
    print("market index done", flush=True)
    early_summers(listings)
    print("2021 and 2022 summers done", flush=True)
    season_2024(listings)
    print(f"wrote {OUT.relative_to(ROOT)}")


def market_index(listings):
    """the monthly index on all sales, and its projection for June to September 2026"""
    ix = MarketIndex().fit(listings)
    months = pd.date_range("2021-01-01", periods=len(ix.levels), freq="MS")
    fitted = pd.DataFrame({"month": months, "level": ix.levels, "kind": "fitted"})
    ahead = pd.date_range(months[-1], periods=5, freq="MS")
    levels = [ix.levels[-1]] + [ix.level([ix.last_m + h])[0] for h in range(1, 5)]
    projected = pd.DataFrame({"month": ahead, "level": levels, "kind": "projected"})
    pd.concat([fitted, projected]).to_csv(OUT / "index.csv", index=False)


def early_summers(listings):
    """run.py backtest covers 2023 to 2025; the spreads chart also shows 2021 and 2022"""
    rows = []
    for year in (2021, 2022):
        start = pd.Timestamp(f"{year}-06-01")
        end = start + pd.DateOffset(months=4)
        train = listings[listings["date_sold"] < start]
        homes = listings[(listings["date_sold"] >= start) & (listings["date_sold"] < end)]
        predicted = models.make_model("off_market").fit(train).predict(data.hide_answers(homes, "off_market"), window_end=end)
        rows.append(pd.DataFrame({"ml_num": homes["ml_num"].to_numpy(), "season": year,
                                  "house_type_name": homes["house_type_name"].to_numpy(),
                                  "sold_price": homes["sold_price"].to_numpy(), "predicted_price": np.round(predicted)}))
    pd.concat(rows).to_csv(OUT / "early_summers.csv", index=False)


def season_2024(listings):
    """everything about the worked example and the tree logic, from the 2024 summer backtest"""
    start = pd.Timestamp("2024-06-01")
    end = start + pd.DateOffset(months=4)
    train = listings[listings["date_sold"] < start]
    homes = listings[(listings["date_sold"] >= start) & (listings["date_sold"] < end)].reset_index(drop=True)
    avm = models.OffMarketAVM().fit(train)
    hidden = data.hide_answers(homes, "off_market").reset_index(drop=True)
    X = avm.builder.transform(hidden)
    level = avm.market_level(hidden, end)
    sold = homes["sold_price"].to_numpy()
    r = int(np.where(homes["ml_num"].to_numpy() == EXAMPLE)[0][0])
    slow = avm.gbms[0]   # one of the four value models: 2,000 trees at rate 0.025, seed 0
    t = avm.train

    # the example home, its 10 nearest earlier sales of the same type, and other sales nearby
    same = np.where(t["house_type_name"].to_numpy() == homes.loc[r, "house_type_name"])[0]
    tree = BallTree(np.radians(t.iloc[same][["latitude", "longitude"]].to_numpy()), metric="haversine")
    _, i = tree.query(np.radians(homes.loc[[r], ["latitude", "longitude"]].to_numpy()), k=10)
    nearest = same[i[0]]
    lat, lon = homes.loc[r, "latitude"], homes.loc[r, "longitude"]
    nearby = np.where((abs(t["latitude"] - lat) < 0.004) & (abs(t["longitude"] - lon) < 0.006))[0]
    others = np.setdiff1d(nearby, nearest)
    pd.concat([
        pd.DataFrame({"role": ["home"], "lat": [lat], "lon": [lon], "ppsf": [np.nan], "miss": [np.nan]}),
        pd.DataFrame({"role": "neighbour", "lat": t["latitude"].to_numpy()[nearest], "lon": t["longitude"].to_numpy()[nearest],
                      "ppsf": np.exp(t["ppsf"].to_numpy()[nearest]), "miss": avm.residual[nearest]}),
        pd.DataFrame({"role": "other", "lat": t["latitude"].to_numpy()[others], "lon": t["longitude"].to_numpy()[others],
                      "ppsf": np.nan, "miss": np.nan}),
    ]).to_csv(OUT / "map.csv", index=False)

    # how much error each possible first cut removes, for two features
    def error_removed(values):
        ok = ~np.isnan(values)
        v, y, w = values[ok], avm.y_train[ok], avm.weights[ok]
        order = np.argsort(v)
        v, y, w = v[order], y[order], w[order]
        cw, cy, cy2 = np.cumsum(w), np.cumsum(w * y), np.cumsum(w * y * y)
        W, Y, Y2 = cw[-1], cy[-1], cy2[-1]
        total = Y2 - Y * Y / W
        out = []
        for q in np.linspace(0.02, 0.98, 49):
            k = int(q * len(v))
            left = cy2[k] - cy[k] ** 2 / cw[k]
            right = (Y2 - cy2[k]) - (Y - cy[k]) ** 2 / (W - cw[k])
            out.append((q, float(v[k]), float(100 * (1 - (left + right) / total))))
        return out
    rows = []
    for name, column in (("Neighbours' value", "knn_val"), ("Size", "lsqft")):
        rows += [{"feature": name, "q": q, "cut": cut, "drop": drop}
                 for q, cut, drop in error_removed(np.exp(avm.X_train[column].to_numpy()))]
    pd.DataFrame(rows).to_csv(OUT / "split.csv", index=False)

    # accuracy as trees are added, and the example home's running value
    marks = {1, 5, 10, 25, 50, 100, 200, 400, 800, 1200, 1600, 2000}
    curve = [(n, mdape(np.exp(p + level), sold)) for n, p in enumerate(slow.staged_predict(X), 1) if n in marks]
    pd.DataFrame(curve, columns=["trees", "mdape"]).to_csv(OUT / "boosting.csv", index=False)
    path = [np.exp(p[0]) for p in slow.staged_predict(X.iloc[[r]])]
    steps = [(0, float(np.exp(slow._baseline_prediction.ravel()[0])))] + \
        [(n, float(path[n - 1])) for n in (1, 2, 3, 5, 10, 25, 50, 100, 200, 500, 1000, 2000)]
    pd.DataFrame(steps, columns=["trees", "value"]).to_csv(OUT / "steps.csv", index=False)

    # one big tree against boosting, all before the local fix
    def codes(frame):
        frame = frame.copy()
        for c in frame.columns:
            if isinstance(frame[c].dtype, pd.CategoricalDtype):
                frame[c] = frame[c].cat.codes
        return frame.fillna(-999)
    value = [g.predict(X) for g in avm.gbms]
    rows = []
    for name, leaf in (("One deep tree", 20), ("One pruned tree", 200)):
        dt = DecisionTreeRegressor(min_samples_leaf=leaf, random_state=0).fit(codes(avm.X_train), avm.y_train, sample_weight=avm.weights)
        rows.append((f"{name} ({dt.get_n_leaves():,} leaves)", mdape(np.exp(dt.predict(codes(X)) + level), sold)))
    rows.append(("Boosting, one model (2,000 trees)", mdape(np.exp(value[0] + level), sold)))
    rows.append(("Four boosted models, averaged", mdape(np.exp(np.mean(value, 0) + level), sold)))
    pd.DataFrame(rows, columns=["model", "mdape"]).to_csv(OUT / "onetree.csv", index=False)

    # permutation importance by feature family
    families = {"Size & rooms": ["lsqft", "area_is_range", "bedrooms", "washrooms", "parking_total", "garage", "lot_front", "lot_depth", "lot_area"],
                "Type & style": ["type", "style"], "Location": ["lat", "lon", "muni", "community"],
                "Same building": ["bld_n", "bld_ppsf", "bld_val"], "Nearest sales": ["knn_ppsf", "knn_km", "knn_val"],
                "Listing text": ["text_score"] + [c for c in X.columns if c.startswith("topic")]}
    base, rng, rows = mdape(np.exp(value[0] + level), sold), np.random.default_rng(0), []
    for name, columns in families.items():
        Z, shuffle = X.copy(), rng.permutation(len(X))
        for c in columns:
            Z[c] = Z[c].iloc[shuffle].to_numpy() if not isinstance(Z[c].dtype, pd.CategoricalDtype) else \
                pd.Categorical(Z[c].to_numpy()[shuffle], categories=Z[c].cat.categories)
        rows.append((name, round(mdape(np.exp(slow.predict(Z) + level), sold) - base, 2)))
    pd.DataFrame(rows, columns=["family", "added"]).to_csv(OUT / "importance.csv", index=False)

    # what the local fix buys, by property type
    avg, fix = np.mean(value, 0) + level, avm.residual_comps(hidden)
    rows = []
    for kind, name in TYPE_NAMES.items():
        m = homes["house_type_name"].to_numpy() == kind
        rows.append((name, mdape(np.exp(avg[m]), sold[m]), mdape(np.exp(avg[m] + fix[m]), sold[m])))
    pd.DataFrame(rows, columns=["type", "without", "with"]).to_csv(OUT / "compsgain.csv", index=False)

    # 100 guessed sale dates for three listing dates, kept if inside June to September
    rows = []
    own = homes.loc[r, "date_listed"]
    for label, listed in ((f"Listed {own:%b} {own.day} (Elmbank house)", own),
                          ("Listed May 20", pd.Timestamp("2024-05-20")), ("Listed Aug 25", pd.Timestamp("2024-08-25"))):
        dates = listed + pd.to_timedelta(avm.dom_quantiles, unit="D")
        kept = dates[(dates >= start) & (dates < end)]
        for month in (6, 7, 8, 9):
            rows.append((label, pd.Timestamp(2024, month, 1).strftime("%B"), int((kept.month == month).sum())))
    pd.DataFrame(rows, columns=["listing", "month", "draws"]).to_csv(OUT / "window.csv", index=False)

    # the seasonal rule: change since May in earlier years, and the median used for 2024
    ix = avm.index
    last = ix.last_m - ix.first_m
    rows = [(str(y), 5, 0.0) for y in range(2021, 2024)] + [("Median (2024 projection)", 5, 0.0)]
    for h in (1, 2, 3, 4):
        for y0 in range(last - 12, -1, -12):
            if y0 + h <= last:
                rows.append((str(2021 + (y0 + ix.first_m) // 12), 5 + h, float(100 * (ix.levels[y0 + h] - ix.levels[y0]))))
        rows.append(("Median (2024 projection)", 5 + h, float(100 * (ix.level([ix.last_m + h])[0] - ix.levels[last]))))
    pd.DataFrame(rows, columns=["year", "month", "change"]).to_csv(OUT / "projex.csv", index=False)


if __name__ == "__main__":
    main()
