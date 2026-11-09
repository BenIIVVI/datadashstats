"""List-aware model (the team's final choice): predict a listed home's sale price from its list price.

Split by property type: condos (condo apartments, condo townhouses, co-ops) and freeholds each get
their own blend of two models.
- A ridge regression on the home's features and list price, predicting the sold price directly.
  Each sale is weighted by 1 / price^3 (rescaled to average 1), so cheap and dear homes count
  more evenly in percentage terms.
- CatBoost on log(sold / list), with community, municipality, style, garage and
  community x property type as categorical inputs, weighted by (median price / price)^1.5.
Blend: 50% CatBoost for condos, 70% CatBoost for freeholds.

Everything is fitted only on the training sales passed to fit(), so validation and the summer
backtests never see a sale from after the homes they predict. Needs `pip install catboost`.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROPERTY_TYPES = ["Condo Apt", "Detached", "Semi-Detached", "Condo Townhouse",
                  "Freehold Townhouse", "Co-Op Apt"]
CONDO_TYPES = ["Condo Apt", "Condo Townhouse", "Co-Op Apt"]
CATEGORICAL = ["municipality", "community", "house_style", "garage_type", "community_property"]


# ---- features ----

def square_feet(house_area):
    """house_area is one number ("622") or a range ("900-1099"): the number, or the middle of the range."""
    parts = house_area.astype(str).str.split("-", expand=True).reindex(columns=[0, 1])
    low = pd.to_numeric(parts[0], errors="coerce")
    high = pd.to_numeric(parts[1], errors="coerce").fillna(low)
    return (low + high) / 2


def basic_features(homes):
    """Size, rooms, parking, location, months since January 2021, property type and list price."""
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
    X["list_price"] = homes["list_price"]
    return X


def catboost_features(homes):
    """The basic features plus neighbourhood, style, garage and lot size for CatBoost."""
    X = basic_features(homes).copy()
    X["municipality"] = homes["municipality"].fillna("Unknown").astype(str)
    X["community"] = homes["community"].fillna("Unknown").astype(str)
    X["house_style"] = homes["house_style"].fillna("Unknown").astype(str)
    X["garage_type"] = homes["garage"].fillna("Unknown").astype(str)
    front = pd.to_numeric(homes["lot_front"], errors="coerce")
    depth = pd.to_numeric(homes["lot_depth"], errors="coerce")
    X["lot_front"] = front
    X["lot_depth"] = depth
    X["lot_area"] = front * depth
    X["community_property"] = (homes["municipality"].fillna("Unknown").astype(str) + "_"
                               + homes["community"].fillna("Unknown").astype(str) + "_"
                               + homes["house_type_name"].fillna("Unknown").astype(str))
    return X


# ---- models ----

class ListPriceRidge:
    """Ridge regression on the basic features, predicting the sold price, weighted by 1 / price^3."""

    def __init__(self, make_features, alpha=1):
        self.make_features = make_features
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))

    def fit(self, train):
        y = train["sold_price"].to_numpy(dtype=float)
        weights = 1 / y ** 3
        weights = weights / np.mean(weights)
        self.model.fit(self.make_features(train), y, ridge__sample_weight=weights)
        return self

    def predict(self, homes):
        return self.model.predict(self.make_features(homes))


class ListPriceRatioCatBoost:
    """CatBoost on log(sold / list); the prediction is list price x exp(predicted log ratio)."""

    def __init__(self, make_features, depth=6, iterations=2000, learning_rate=0.05, power=1.5, l2_leaf_reg=20):
        self.make_features = make_features
        self.power = power
        self.model = CatBoostRegressor(iterations=iterations, depth=depth, learning_rate=learning_rate,
                                       l2_leaf_reg=l2_leaf_reg, loss_function="RMSE", verbose=False,
                                       random_seed=42)

    def fit(self, train):
        list_price = train["list_price"].to_numpy(dtype=float)
        sold_price = train["sold_price"].to_numpy(dtype=float)
        y = np.log(sold_price / list_price)
        weights = (np.median(sold_price) / sold_price) ** self.power
        weights /= weights.mean()
        self.model.fit(self.make_features(train), y, sample_weight=weights, cat_features=CATEGORICAL)
        return self

    def predict(self, homes):
        return np.exp(self.model.predict(self.make_features(homes))) * homes["list_price"].to_numpy(dtype=float)


class ListPriceEnsemble:
    """Blend: (1 - w) x ridge + w x CatBoost."""

    def __init__(self, make_features, catboost_weight=0.6):
        self.catboost_weight = catboost_weight
        self.ridge = ListPriceRidge(make_features, alpha=100)
        self.catboost = ListPriceRatioCatBoost(catboost_features, depth=6, iterations=2000,
                                               learning_rate=0.05, power=1.5, l2_leaf_reg=20)

    def fit(self, train):
        self.ridge.fit(train)
        self.catboost.fit(train)
        return self

    def predict(self, homes):
        return ((1 - self.catboost_weight) * self.ridge.predict(homes)
                + self.catboost_weight * self.catboost.predict(homes))


class SplitPropertyEnsemble:
    """Separate blends for condos (50% CatBoost) and freeholds (70% CatBoost)."""

    def __init__(self, make_features=basic_features):
        self.condo = ListPriceEnsemble(make_features, catboost_weight=0.5)
        self.freehold = ListPriceEnsemble(make_features, catboost_weight=0.7)

    def fit(self, train):
        condo = train["house_type_name"].isin(CONDO_TYPES)
        self.condo.fit(train.loc[condo])
        self.freehold.fit(train.loc[~condo])
        return self

    def predict(self, homes, **_):
        """**_ accepts run.py's window_end argument, which this model does not use."""
        condo = homes["house_type_name"].isin(CONDO_TYPES)
        predictions = pd.Series(index=homes.index, dtype=float)
        if condo.any():
            predictions.loc[condo] = self.condo.predict(homes.loc[condo])
        if (~condo).any():
            predictions.loc[~condo] = self.freehold.predict(homes.loc[~condo])
        return predictions.to_numpy()
