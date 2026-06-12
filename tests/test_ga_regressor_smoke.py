import numpy as np
import pandas as pd
from sklearn.base import clone

from model.estimator import GARegressor


def _toy_data(n_rows=32):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n_rows),
            "x2": rng.normal(size=n_rows),
        }
    )
    loc = pd.DataFrame(
        {
            "coord_x": rng.random(n_rows),
            "coord_y": rng.random(n_rows),
        }
    )
    y = pd.DataFrame(
        {
            "target": 2 * X["x1"] - X["x2"] + 0.1 * rng.normal(size=n_rows),
        }
    )
    return X, loc, y


def test_ga_regressor_is_cloneable():
    model = GARegressor(epochs=1, device="cpu", verbose=False)
    cloned = clone(model)

    assert cloned.get_params()["epochs"] == 1
    assert cloned.get_params()["device"] == "cpu"
    assert cloned.get_params()["normalize_spatial"] is True


def test_ga_regressor_fit_predict_smoke():
    X, loc, y = _toy_data()
    model = GARegressor(
        attn_variant="MCPA",
        d_model=32,
        n_attn_layer=1,
        idu_points=1,
        seq_len=16,
        epochs=1,
        batch_size=8,
        device="cpu",
        random_state=0,
        verbose=False,
    )

    returned = model.fit(X, loc, y)
    pred = model.predict(X.iloc[:8], loc.iloc[:8], n_estimate=2, verbose=False)

    assert returned is model
    assert pred.shape[0] == 8
    assert np.isfinite(pred.numpy()).all()


def test_ga_regressor_normalizes_large_spatial_coordinates():
    X, loc, y = _toy_data()
    loc = loc * 100000 + 500000
    model = GARegressor(
        attn_variant="MCPA",
        d_model=32,
        n_attn_layer=1,
        idu_points=1,
        seq_len=16,
        epochs=1,
        batch_size=8,
        device="cpu",
        random_state=0,
        verbose=False,
    )

    model.fit(X, loc, y)
    spa_cols_id = model.tab_sampler.spa_cols_id
    normalized_loc = model.tab_sampler.context_pool[:, spa_cols_id]

    assert normalized_loc.min() >= 0.0
    assert normalized_loc.max() <= 1.0


def test_ga_regressor_can_disable_spatial_normalization():
    X, loc, y = _toy_data()
    loc = loc * 100000 + 500000
    model = GARegressor(
        attn_variant="MCPA",
        d_model=32,
        n_attn_layer=1,
        idu_points=1,
        seq_len=16,
        epochs=1,
        batch_size=8,
        device="cpu",
        normalize_spatial=False,
        random_state=0,
        verbose=False,
    )

    model.fit(X, loc, y)
    spa_cols_id = model.tab_sampler.spa_cols_id
    raw_loc = model.tab_sampler.context_pool[:, spa_cols_id]

    assert raw_loc.max() > 1000.0
