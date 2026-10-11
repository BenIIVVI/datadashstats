"""Features: everything a model knows about a home, built only from sales before it.

Three building blocks, each fitted on the training sales and then applied to new homes:

- MarketIndex: a monthly price index from a hedonic regression (log price on community x type,
  size, rooms and month). Prices are "deflated" by it, so the tree model learns what a home is
  worth in constant dollars, and the index carries the market level, including a seasonal
  projection past the last month of data.
- location: past sales in the same building (condos share coordinates) and the k nearest
  earlier sales of the same type, both as deflated log price per square foot.
- text: a ridge regression on TF-IDF of the remarks, plus 30 SVD topics, which pick up
  condition ("fully renovated", "original finishes", "sold as-is").

Rule followed everywhere: a training row only sees sales strictly before it, and a home being
predicted only sees the training sales.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]
CATEGORICAL = ["type", "style", "muni", "community"]


def square_feet(house_area):
    """house_area is text, either one number ("622") or a range ("900-1099").
    Return a number: the number itself, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def month_index(dates):
    """Whole months since January 2021 (January 2021 = 0)."""
    dates = pd.DatetimeIndex(dates)
    return np.asarray((dates.year - 2021) * 12 + dates.month - 1)


def month_start(m):
    return pd.Timestamp(2021, 1, 1) + pd.DateOffset(months=int(m))


# ---------------------------------------------------------------- market index

class MarketIndex:
    season_weight = 1.0   # share of the past seasonal change applied when projecting

    """Monthly log price level, net of what is sold (location, type, size).

    Fitted by ridge regression of log price on community x type dummies, log square feet,
    bedrooms, washrooms, parking and one dummy per month of sale. Months after the data end
    are projected as: last level + the median change over the same calendar months in
    earlier years (the summer slide that repeats every year).
    """

    def fit(self, train):
        one_hot = lambda codes, n: sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                                                 shape=(len(codes), n))
        groups = pd.Categorical(train["community"] + "|" + train["house_type_name"])
        size = np.log(square_feet(train["house_area"]).fillna(800).to_numpy())
        numbers = np.column_stack([size, train["bedrooms"], train["washrooms"], train["parking_total"]])
        numbers = (numbers - numbers.mean(0)) / (numbers.std(0) + 1e-9)
        months = month_index(train["date_sold"])
        self.first_m, self.last_m = int(months.min()), int(months.max())
        n_months = self.last_m - self.first_m + 1
        X = sp.hstack([one_hot(groups.codes, len(groups.categories)), sp.csr_matrix(numbers),
                       one_hot(months - self.first_m, n_months)]).tocsr()
        coef = Ridge(alpha=1.0).fit(X, np.log(train["sold_price"])).coef_[-n_months:]
        self.levels = coef - coef.mean()
        return self

    @classmethod
    def from_series(cls, series):
        """An index from an existing monthly series (month number -> value), e.g. sale-to-list."""
        out = cls()
        full = series.reindex(range(int(series.index.min()), int(series.index.max()) + 1)).interpolate()
        out.first_m, out.last_m, out.levels = int(full.index.min()), int(full.index.max()), full.to_numpy()
        return out

    def level(self, months):
        """Index for months inside the data, or projected for months after it."""
        months = np.asarray(months)
        out = np.empty(len(months))
        for i, m in enumerate(months):
            out[i] = self._one(int(m))
        return out

    def _one(self, m):
        m = min(max(m, self.first_m), self.last_m + 12)
        series = self.levels
        if m <= self.last_m:
            return series[m - self.first_m]
        h, last = m - self.last_m, self.last_m - self.first_m
        changes = [series[y + h] - series[y] for y in range(last - 12, -1, -12) if y + h <= last]
        return series[last] + self.season_weight * (np.median(changes) if changes else 0.0)

    def expected(self, date_listed, window_start, window_end, dom_quantiles):
        """Average index over the dates a home could plausibly sell: its listing date plus a
        days-on-market draw from past sales, kept only if inside [window_start, window_end)."""
        cache, out = {}, np.empty(len(date_listed))
        for i, listed in enumerate(pd.DatetimeIndex(date_listed)):
            if listed not in cache:
                dates = listed + pd.to_timedelta(dom_quantiles, unit="D")
                dates = dates[(dates >= window_start) & (dates < window_end)]
                if len(dates) == 0:   # a slow sale: the first day it could have sold
                    dates = pd.DatetimeIndex([min(max(listed, window_start), window_end - pd.Timedelta(days=1))])
                cache[listed] = self.level(month_index(dates)).mean()
            out[i] = cache[listed]
        return out


# ---------------------------------------------------------------- feature builder

def _location_key(homes):
    return homes["latitude"].round(6).astype(str) + "," + homes["longitude"].round(6).astype(str)


class FeatureBuilder:
    """Fit on training sales (with deflated log price per sq ft already computed), then build
    the same feature table for the training rows (strictly-earlier sales only) or new homes."""

    def __init__(self, k=10, svd_topics=30):
        self.k, self.svd_topics = k, svd_topics

    def fit(self, train):
        """train needs columns ppsf (deflated log price per sq ft) and lsqft."""
        self.train = train
        self.styles = sorted(train["house_style"].unique())
        self.munis = sorted(train["municipality"].unique())
        self.communities = sorted(train["community"].unique())
        # text: ridge on what location and type do not explain, out-of-fold for training rows
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=min(20, max(1, len(train) // 100)),
                                     max_features=60000, sublinear_tf=True)
        X = self.tfidf.fit_transform(train["remarks"])
        target = (train["ppsf"] - train.groupby(["community", "house_type_name"])["ppsf"].transform("mean")).to_numpy()
        oof = np.zeros(len(train))
        for fit_rows, held in KFold(5, shuffle=True, random_state=0).split(X):
            oof[held] = Ridge(alpha=3.0).fit(X[fit_rows], target[fit_rows]).predict(X[held])
        self.text_oof = oof
        self.ridge = Ridge(alpha=3.0).fit(X, target)
        n_topics = min(self.svd_topics, X.shape[1] - 1)
        self.svd = TruncatedSVD(n_topics, random_state=0).fit(X) if n_topics >= 2 else None
        self.train_X_text = X
        return self

    def training_features(self):
        return self._build(self.train, is_train=True)

    def transform(self, homes):
        return self._build(homes, is_train=False)

    # ---- pieces
    def _building(self, homes, is_train):
        t = self.train
        key_t = _location_key(t)
        if is_train:   # t is sorted by date_sold: cumulative stats exclude the row itself
            g = t.assign(_k=key_t).groupby("_k")["ppsf"]
            count = g.cumcount().to_numpy()
            total = (g.cumsum() - t["ppsf"]).to_numpy()
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
            return count, mean
        stats = t.assign(_k=key_t).groupby("_k")["ppsf"].agg(["mean", "size"])
        key_h = _location_key(homes)
        return key_h.map(stats["size"]).fillna(0).to_numpy(), key_h.map(stats["mean"]).to_numpy()

    def _nearest(self, homes, is_train):
        t, k = self.train, self.k
        median = np.full(len(homes), np.nan)
        km = np.full(len(homes), np.nan)
        for kind in PROPERTY_TYPES:
            same = t[t["house_type_name"] == kind]
            rows = np.where(homes["house_type_name"].to_numpy() == kind)[0]
            if len(rows) == 0 or len(same) < k:
                continue
            xy_t = np.radians(same[["latitude", "longitude"]].to_numpy())
            xy_h = np.radians(homes.iloc[rows][["latitude", "longitude"]].to_numpy())
            ppsf = same["ppsf"].to_numpy()
            if not is_train:
                d, i = BallTree(xy_t, metric="haversine").query(xy_h, k=k)
                median[rows], km[rows] = np.median(ppsf[i], 1), d.mean(1) * 6371
                continue
            # training rows: neighbours sold in earlier months only
            m_t = month_index(same["date_sold"])
            m_h = month_index(homes.iloc[rows]["date_sold"])
            for m in np.unique(m_h):
                prior = m_t < m
                if prior.sum() < k:
                    continue
                pick = m_h == m
                d, i = BallTree(xy_t[prior], metric="haversine").query(xy_h[pick], k=k)
                median[rows[pick]], km[rows[pick]] = np.median(ppsf[prior][i], 1), d.mean(1) * 6371
        return median, km

    def _build(self, h, is_train):
        X = pd.DataFrame(index=h.index)
        X["lsqft"] = np.log(square_feet(h["house_area"]))
        X["area_is_range"] = h["house_area"].astype(str).str.contains("-").astype(int)
        for c in ["bedrooms", "washrooms", "parking_total", "garage", "lot_front", "lot_depth"]:
            X[c] = h[c].astype(float)
        X["lot_area"] = X["lot_front"] * X["lot_depth"]
        X["lat"], X["lon"] = h["latitude"], h["longitude"]
        X["type"] = pd.Categorical(h["house_type_name"], categories=PROPERTY_TYPES)
        X["style"] = pd.Categorical(h["house_style"], categories=self.styles)
        X["muni"] = pd.Categorical(h["municipality"], categories=self.munis)
        X["community"] = pd.Categorical(h["community"], categories=self.communities)
        X["bld_n"], X["bld_ppsf"] = self._building(h, is_train)
        X["knn_ppsf"], X["knn_km"] = self._nearest(h, is_train)
        X["bld_val"] = X["bld_ppsf"] + X["lsqft"]
        X["knn_val"] = X["knn_ppsf"] + X["lsqft"]
        text = self.train_X_text if is_train else self.tfidf.transform(h["remarks"])
        X["text_score"] = self.text_oof if is_train else self.ridge.predict(text)
        if self.svd is not None:
            topics = self.svd.transform(text)
            for j in range(topics.shape[1]):
                X[f"topic{j}"] = topics[:, j]
        return X


# ---------------------------------------------------------------- starter features (kept for FeatureModel)

def off_market_features(homes):
    """The starter's simple features, kept so FeatureModel still works as a comparison."""
    X = pd.DataFrame(index=homes.index)
    X["square_feet"] = square_feet(homes["house_area"])
    X["bedrooms"] = homes["bedrooms"]
    X["washrooms"] = homes["washrooms"]
    X["parking"] = homes["parking_total"]
    X["latitude"] = homes["latitude"]
    X["longitude"] = homes["longitude"]
    X["months"] = (homes["date_listed"] - pd.Timestamp("2021-01-01")).dt.days / 30.44
    for kind in PROPERTY_TYPES:
        X["is_" + kind] = (homes["house_type_name"] == kind).astype(int)
    return X


def list_aware_features(homes):
    X = off_market_features(homes)
    X["list_price"] = homes["list_price"]
    return X
