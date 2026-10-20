"""Models. Every model has the same two steps:

    model.fit(train)                     train = past sales, with sold_price and date_sold
    model.predict(homes, window_end)     homes = homes to value (test-file columns only);
                                         window_end = first day after the prediction window

The prediction window starts the day after the last training sale. For the real test that is
June 1 to October 1 2026; in validation it is whatever months are being predicted.

OffMarketAVM   no list price. Gradient-boosted trees on deflated log price, plus the market index.
ListAwareModel predicts log(sold / list) from the gap between the list price and the off-market
               value, how long the listing has already been up, and the recent sale-to-list level.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

import features
from features import CATEGORICAL, FeatureBuilder, MarketIndex, month_index, month_start

GBM_PARAMS = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=False, random_state=0)
# the off-market value model itself: slower learning, smaller leaves, 30% of features tried at
# each split, averaged over two seeds; chosen on the 2023 and 2024 backtests, where it cut MdAPE
# by 0.07 and 0.22 points and MAPE by 0.2 (with residual comps). The out-of-fold fits behind
# residual comps and the list-aware model keep GBM_PARAMS: a tuned out-of-fold model was no better.
VALUE_PARAMS = dict(learning_rate=0.025, max_iter=2000, min_samples_leaf=20, max_features=0.3)
VALUE_SEEDS = (0, 1)
RECENCY_HALFLIFE_MONTHS = 12   # a sale 12 months before the data end counts half as much
DEFAULT_HORIZON_MONTHS = 4     # June to September
# residual comps: shift each off-market prediction by the average out-of-fold miss of the 10
# nearest earlier sales of the same type, shrunk toward zero (sum / (10 + 20)); chosen on the
# 2023 and 2024 backtests, where it cut MdAPE by 0.14 and 0.21 points, mostly on houses
RESIDUAL_K, RESIDUAL_SHRINK = 10, 20


def make_model(track):
    """Return a fresh, untrained model for "off_market" or "list_aware"."""
    if track == "list_aware":
        return ListAwareModel()
    return OffMarketAVM()


def _gbm(columns, **overrides):
    return HistGradientBoostingRegressor(categorical_features=[c for c in columns if c in CATEGORICAL],
                                         **{**GBM_PARAMS, **overrides})


# the list-aware model reuses the off-market model fitted on the same training sales
_FITTED = {}


def _train_key(train):
    return (len(train), str(train["date_sold"].max()), str(train["ml_num"].iloc[0]), str(train["ml_num"].iloc[-1]))


class OffMarketAVM:
    def fit(self, train):
        key = _train_key(train)
        if key in _FITTED:
            self.__dict__.update(_FITTED[key].__dict__)
            return self
        t = train.sort_values(["date_sold", "ml_num"]).reset_index(drop=True)
        self.cutoff = t["date_sold"].max().normalize() + pd.Timedelta(days=1)
        self.index = MarketIndex().fit(t)
        self.dom_quantiles = np.quantile(t["days_on_market"], np.linspace(0.005, 0.995, 100))
        months = month_index(t["date_sold"])
        t["lsqft"] = np.log(features.square_feet(t["house_area"]))
        t["ystar"] = np.log(t["sold_price"]) - self.index.level(months)   # constant-dollar log price
        t["ppsf"] = t["ystar"] - t["lsqft"]
        self.builder = FeatureBuilder().fit(t)
        self.X_train = self.builder.training_features()
        self.y_train = t["ystar"].to_numpy()
        self.weights = 0.5 ** ((months.max() - months) / RECENCY_HALFLIFE_MONTHS)
        self.train = t
        self.gbms = [_gbm(self.X_train.columns, **VALUE_PARAMS, random_state=seed)
                     .fit(self.X_train, self.y_train, sample_weight=self.weights) for seed in VALUE_SEEDS]
        # out-of-fold value for every training sale: what the model guesses without having seen it
        self.oof = np.zeros(len(t))
        for fit_rows, held in KFold(5, shuffle=True, random_state=1).split(self.X_train):
            g = _gbm(self.X_train.columns).fit(self.X_train.iloc[fit_rows], self.y_train[fit_rows],
                                               sample_weight=self.weights[fit_rows])
            self.oof[held] = g.predict(self.X_train.iloc[held])
        self.residual = self.y_train - self.oof   # how far each sale landed from the model's guess
        _FITTED.clear()
        _FITTED[key] = self
        return self

    def window_end(self, window_end):
        return pd.Timestamp(window_end) if window_end is not None else \
            self.cutoff + pd.DateOffset(months=DEFAULT_HORIZON_MONTHS)

    def market_level(self, homes, window_end=None):
        return self.index.expected(homes["date_listed"], self.cutoff, self.window_end(window_end), self.dom_quantiles)

    def residual_comps(self, homes):
        """shrunk mean out-of-fold miss of the nearest training sales of the same type"""
        t, k, out = self.train, RESIDUAL_K, np.zeros(len(homes))
        for kind in features.PROPERTY_TYPES:
            same = np.where(t["house_type_name"].to_numpy() == kind)[0]
            rows = np.where(homes["house_type_name"].to_numpy() == kind)[0]
            if len(rows) == 0 or len(same) < k:
                continue
            tree = BallTree(np.radians(t.iloc[same][["latitude", "longitude"]].to_numpy()), metric="haversine")
            _, i = tree.query(np.radians(homes.iloc[rows][["latitude", "longitude"]].to_numpy()), k=k)
            out[rows] = self.residual[same][i].sum(1) / (k + RESIDUAL_SHRINK)
        return out

    def predict_log(self, homes, window_end=None, comps=True):
        homes = homes.reset_index(drop=True)
        X = self.builder.transform(homes)
        value = np.mean([g.predict(X) for g in self.gbms], 0) + self.market_level(homes, window_end)
        if comps:
            value = value + self.residual_comps(homes)
        return value, X

    def predict(self, homes, window_end=None):
        return np.exp(self.predict_log(homes, window_end)[0])


class ListAwareModel:
    """Target: log(sold / list), net of a monthly sale-to-list level that is projected past the
    data with the same seasonal rule as the price index (ratios slip from spring to summer). Key inputs: the gap between log list price and the off-market
    value, days already on the market when the prediction is made, and the recent sale-to-list
    level. Training rows simulate a prediction made 0 to 3 months before the sale month, the
    same spread of horizons as the June to September test."""

    def fit(self, train):
        self.avm = OffMarketAVM().fit(train)
        t, X = self.avm.train, self.avm.X_train
        oof = self.avm.oof   # out-of-fold value, so the gap is not artificially small
        sale_m = month_index(t["date_sold"])
        value = oof + self.avm.index.level(sale_m)
        # monthly median log sale-to-list, to describe how hot the market was before a prediction
        log_ratio = np.log(t["sold_price"] / t["list_price"]).to_numpy()
        self.ratio_by_month = pd.Series(log_ratio).groupby(sale_m).median()
        self.ratio_index = MarketIndex.from_series(self.ratio_by_month)
        # the summer slip in sale-to-list has shrunk each year since 2021, so the median past
        # change overstates it; apply half (chosen on the 2023 and 2024 backtests)
        self.ratio_index.season_weight = 0.5
        target = log_ratio - self.ratio_index.level(sale_m)   # sale-to-list net of that month's level
        rng = np.random.default_rng(0)
        cutoff_m = sale_m - rng.integers(0, DEFAULT_HORIZON_MONTHS, len(t))
        cutoff = pd.DatetimeIndex([month_start(m) for m in cutoff_m])
        Z = self._extra(X, t, value, cutoff, cutoff_m)
        self.columns = Z.columns
        self.gbm = _gbm(Z.columns, max_iter=600).fit(Z, target, sample_weight=self.avm.weights)
        return self

    def _ratio_level(self, cutoff_m):
        """mean of the monthly median log sale-to-list over the 3 months before the prediction"""
        s, out = self.ratio_by_month, {}
        for m in np.unique(cutoff_m):
            window = s[(s.index >= m - 3) & (s.index < m)]
            out[m] = window.mean() if len(window) else s.iloc[-3:].mean()
        return np.array([out[m] for m in cutoff_m])

    def _extra(self, X, homes, value, cutoff, cutoff_m):
        Z = X.reset_index(drop=True).copy()
        log_list = np.log(homes["list_price"].to_numpy())
        Z["log_list"] = log_list
        Z["list_vs_value"] = log_list - value
        Z["days_listed_so_far"] = np.maximum((cutoff - pd.DatetimeIndex(homes["date_listed"])).days, 0)
        Z["listed_month_of_year"] = pd.DatetimeIndex(homes["date_listed"]).month
        Z["recent_sale_to_list"] = self._ratio_level(cutoff_m)
        return Z

    def predict(self, homes, window_end=None):
        homes = homes.reset_index(drop=True)
        value, X = self.avm.predict_log(homes, window_end, comps=False)
        cutoff = pd.DatetimeIndex([self.avm.cutoff] * len(homes))
        cutoff_m = np.full(len(homes), month_index([self.avm.cutoff])[0])
        Z = self._extra(X, homes, value, cutoff, cutoff_m)
        level = self.ratio_index.expected(homes["date_listed"], self.avm.cutoff, self.avm.window_end(window_end),
                                          self.avm.dom_quantiles)
        return homes["list_price"].to_numpy() * np.exp(self.gbm.predict(Z[self.columns]) + level)


# ---------------------------------------------------------------- starter baselines, for comparison

class ListPriceBaseline:
    """List-aware baseline: every home sells for exactly its list price."""

    def fit(self, train):
        return self

    def predict(self, homes, window_end=None):
        return homes["list_price"].to_numpy(dtype=float)


class NearestSalesBaseline:
    """Off-market baseline: median sold price of the k nearest earlier sales of the same type."""

    def __init__(self, k=10):
        self.k = k

    def fit(self, train):
        columns = ["ml_num", "house_type_name", "date_sold", "latitude", "longitude", "sold_price"]
        self.sales = train[columns].sort_values(["date_sold", "ml_num"], ascending=[False, True])
        self.sales_by_type = dict(list(self.sales.groupby("house_type_name")))
        return self

    def predict(self, homes, window_end=None):
        return np.array([self.predict_one(home) for _, home in homes.iterrows()])

    def predict_one(self, home):
        sales = self.sales_by_type.get(home["house_type_name"], self.sales)
        earlier = sales[sales["date_sold"] < home["date_listed"]]
        if len(earlier) == 0:
            earlier = self.sales[self.sales["date_sold"] < home["date_listed"]]
        dx = (earlier["longitude"].to_numpy() - home["longitude"]) * 0.72
        dy = earlier["latitude"].to_numpy() - home["latitude"]
        nearest = np.argsort(np.sqrt(dx ** 2 + dy ** 2), kind="stable")[: self.k]
        return float(np.median(earlier["sold_price"].to_numpy()[nearest]))


class FeatureModel:
    """The starter's gradient-boosted model on simple features, kept as a comparison."""

    def __init__(self, make_features):
        self.make_features = make_features
        self.model = HistGradientBoostingRegressor(random_state=0)

    def fit(self, train):
        self.model.fit(self.make_features(train), np.log(train["sold_price"]))
        return self

    def predict(self, homes, window_end=None):
        return np.exp(self.model.predict(self.make_features(homes)))
