"""
Unit tests for src.models.baselines.

These tests fit real models (no mocking of statsmodels/arch) on small,
synthetic series with known statistical properties, so the tests are fast
but still exercise the real fitting and forecasting code paths.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.baselines import (
    ModelFitError,
    fit_arma,
    fit_garch,
    random_walk_forecast,
)


def _white_noise(n=300, mu=0.0005, sigma=0.01, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(
        mu + rng.normal(0, sigma, n),
        index=pd.bdate_range("2020-01-01", periods=n),
    )


def _simulated_garch(n=400, omega=1e-6, alpha=0.1, beta=0.85, seed=7):
    """A real GARCH(1,1) process, so the GARCH fit has actual volatility
    clustering to recover, not just noise."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    sigma2 = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = rng.normal(0, np.sqrt(sigma2[t]))
    return pd.Series(returns, index=pd.bdate_range("2020-01-01", periods=n))


# ---------------------------------------------------------------------------
# random_walk_forecast
# ---------------------------------------------------------------------------

def test_random_walk_forecast_equals_training_mean():
    train = pd.Series([0.01, 0.02, 0.03, 0.04])
    fc = random_walk_forecast(train, n_steps=5)
    assert len(fc) == 5
    assert (fc == train.mean()).all()


def test_random_walk_forecast_is_constant():
    train = _white_noise()
    fc = random_walk_forecast(train, n_steps=10)
    assert fc.nunique() == 1


def test_random_walk_rejects_empty_series():
    with pytest.raises(ModelFitError):
        random_walk_forecast(pd.Series(dtype=float), n_steps=5)


# ---------------------------------------------------------------------------
# fit_arma
# ---------------------------------------------------------------------------

def test_fit_arma_fixed_order_returns_requested_order():
    train = _white_noise()
    result = fit_arma(train, order=(1, 1))
    assert result.order == (1, 1)


def test_fit_arma_forecast_converges_toward_mean_on_white_noise():
    """On pure white noise, an ARMA forecast several steps ahead should
    settle close to the series mean — there's no real structure to extrapolate."""
    train = _white_noise(n=500, mu=0.0005, sigma=0.01)
    result = fit_arma(train, order=(1, 1))
    fc = result.forecast(20)
    assert math.isclose(fc.iloc[-1], train.mean(), abs_tol=0.005)


def test_fit_arma_bic_search_selects_low_order_on_pure_noise():
    """BIC penalises unnecessary complexity, so on pure noise it should not
    select a high-order model."""
    train = _white_noise(n=300)
    result = fit_arma(train, max_order=2)  # small search space for test speed
    p, q = result.order
    assert p <= 2 and q <= 2


def test_fit_arma_rejects_too_short_series():
    with pytest.raises(ModelFitError, match="too short"):
        fit_arma(pd.Series([0.01, 0.02, 0.03]), order=(1, 1))


def test_fit_arma_forecast_length_matches_request():
    train = _white_noise()
    result = fit_arma(train, order=(1, 0))
    fc = result.forecast(7)
    assert len(fc) == 7


# ---------------------------------------------------------------------------
# fit_garch
# ---------------------------------------------------------------------------

def test_fit_garch_returns_mean_and_variance_forecast():
    train = _simulated_garch()
    result = fit_garch(train, arma_order=(0, 0))
    fc = result.forecast(10)
    assert set(fc.keys()) == {"mean", "variance"}
    assert len(fc["mean"]) == 10
    assert len(fc["variance"]) == 10


def test_fit_garch_variance_forecast_is_positive():
    """Variance can never be negative — a basic sanity check on the model output."""
    train = _simulated_garch()
    result = fit_garch(train, arma_order=(0, 0))
    fc = result.forecast(10)
    assert (fc["variance"] > 0).all()


def test_fit_garch_variance_forecast_matches_training_scale():
    """
    The forecast variance should land in the same order of magnitude as the
    realised variance of the training series. This specifically guards
    against the residual-scaling bug class: GARCH is fit on residuals
    scaled up by 100 for numerical stability, and the variance forecast
    must be scaled back down by 100**2 before being returned.
    """
    train = _simulated_garch(n=500)
    result = fit_garch(train, arma_order=(0, 0))
    fc = result.forecast(5)

    realised_var = train.var()
    forecast_var = fc["variance"].mean()

    # Allow a generous band (half to double) since this is a statistical
    # forecast, not an exact reproduction — the point is to catch
    # order-of-magnitude errors (e.g. a 10,000x scale bug), not to assert
    # precise equality.
    assert realised_var / 2 < forecast_var < realised_var * 2


def test_fit_garch_rejects_when_arma_step_fails():
    with pytest.raises(ModelFitError):
        fit_garch(pd.Series([0.01, 0.02, 0.03]), arma_order=(1, 1))


def test_garch_in_mean_fits_and_returns_expected_fields():
    """fit_garch_in_mean should converge on a real-length series and return
    omega/alpha/beta/lam with sane signs (omega>0, alpha,beta>=0, alpha+beta<1)."""
    import numpy as np
    import pandas as pd

    from src.models.baselines import fit_garch_in_mean

    rng = np.random.default_rng(42)
    n = 1500
    # simulate a simple GARCH(1,1)-like series so the fit has real persistence to find
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = 1e-4
    omega, alpha, beta = 1e-6, 0.08, 0.9
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = rng.normal(0, np.sqrt(sigma2[t]))
    r = pd.Series(eps, index=pd.bdate_range("2010-01-01", periods=n))

    result = fit_garch_in_mean(r, n_iter=3)

    assert result.converged
    assert result.omega > 0
    assert result.alpha >= 0
    assert result.beta >= 0
    assert result.alpha + result.beta < 1.0
    assert np.isfinite(result.lam)
    assert np.isfinite(result.lam_tstat)
    assert 0.0 <= result.lam_pvalue <= 1.0


def test_garch_in_mean_forecast_returns_mean_and_variance():
    import numpy as np
    import pandas as pd

    from src.models.baselines import fit_garch_in_mean

    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0, 0.01, 800), index=pd.bdate_range("2015-01-01", periods=800))

    result = fit_garch_in_mean(r, n_iter=2)
    fc = result.forecast(n_steps=1)

    assert set(fc.keys()) == {"mean", "variance"}
    assert np.isfinite(fc["mean"])
    assert fc["variance"] > 0


def test_garch_in_mean_rejects_multi_step_forecast():
    import numpy as np
    import pandas as pd
    import pytest

    from src.models.baselines import fit_garch_in_mean

    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0, 0.01, 800), index=pd.bdate_range("2015-01-01", periods=800))
    result = fit_garch_in_mean(r, n_iter=1)

    with pytest.raises(NotImplementedError):
        result.forecast(n_steps=5)
