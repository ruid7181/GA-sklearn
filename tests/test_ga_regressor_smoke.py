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
