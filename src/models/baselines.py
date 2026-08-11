"""
Classical baseline forecasting models: random walk, ARMA, and GARCH.

These are the "must beat" baselines from the proposal (Section 5.2) that the
deep learning models (LSTM, GRU, TFT) are compared against in RQ1.

GARCH implementation note
--------------------------
The proposal specifies an "ARMA mean equation with GARCH(1,1) conditional
variance" (Section 5.2, model 3). The `arch` package's built-in `arch_model`
only supports AR-type mean equations (Constant, Zero, LS, AR, ARX, HAR,
HARX) — there is no native MA term. To get a true ARMA(p,q) mean with
GARCH(1,1) variance, this module uses the standard two-step approach:

    1. Fit ARMA(p, q) via statsmodels on the raw return series.
    2. Fit GARCH(1,1) via `arch` on the ARMA residuals (mean="Zero", since
       the mean structure was already captured in step 1).

This is a well-established way to combine the two model families and is
what the proposal's "ARMA mean equation with GARCH(1,1) conditional
variance" wording implies, given the limitations of the GARCH-fitting
library used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


class ModelFitError(ValueError):
    """Raised when a model fails to fit or converge."""


# ---------------------------------------------------------------------------
# Random walk (the proposal's "null that any useful approach should beat")
# ---------------------------------------------------------------------------

def random_walk_forecast(train: pd.Series, n_steps: int) -> pd.Series:
    """
    Random walk with drift: forecast every future step as the in-sample
    mean return (proposal Section 5.2, model 1).

    Parameters
    ----------
    train : pd.Series
        Training log-returns.
    n_steps : int
        Number of steps to forecast.

    Returns
    -------
    pd.Series
        Constant forecast (the training mean), length n_steps, with a
        plain integer index (0..n_steps-1) — the caller is expected to
        assign real dates, since this function has no knowledge of the
        test period's calendar.
    """
    if len(train) == 0:
        raise ModelFitError("Cannot compute a random walk forecast from an empty series.")
    drift = train.mean()
    return pd.Series([drift] * n_steps, index=range(n_steps))


# ---------------------------------------------------------------------------
# ARMA
# ---------------------------------------------------------------------------

@dataclass
class ARMAResult:
    """Wraps a fitted ARMA model with the pieces downstream code needs."""

    order: tuple[int, int]
    fitted_model: object  # statsmodels ARIMAResultsWrapper
    resid: pd.Series

    def forecast(self, n_steps: int) -> pd.Series:
        """Point forecast for the next n_steps, with a plain integer index."""
        fc = self.fitted_model.forecast(steps=n_steps)
        return pd.Series(fc.values, index=range(n_steps))


def fit_arma(
    train: pd.Series,
    order: tuple[int, int] | None = None,
    max_order: int = 5,
) -> ARMAResult:
    """
    Fit an ARMA(p, q) model to a training return series.

    Parameters
    ----------
    train : pd.Series
        Training log-returns. Assumed already stationary (the proposal
        treats log-returns as approximately stationary — see Section 5.2
        and notebook 03's ADF test).
    order : tuple[int, int], optional
        Fixed (p, q) order. If None, the order is selected by BIC over
        p, q in {0, ..., max_order}, per proposal Section 5.2.
    max_order : int
        Upper bound for the BIC search when `order` is not given.

    Returns
    -------
    ARMAResult

    Raises
    ------
    ModelFitError
        If no candidate order converges, or the series is too short to fit.
    """
    if len(train) < 10:
        raise ModelFitError(
            f"Training series has only {len(train)} observations; too short to fit ARMA."
        )

    if order is not None:
        return _fit_arma_single(train, order)

    best_bic = np.inf
    best_result = None
    for p, q in product(range(max_order + 1), range(max_order + 1)):
        if p == 0 and q == 0:
            continue  # a pure-constant model isn't useful as an ARMA baseline
        try:
            candidate = _fit_arma_single(train, (p, q))
        except ModelFitError:
            continue
        if candidate.fitted_model.bic < best_bic:
            best_bic = candidate.fitted_model.bic
            best_result = candidate

    if best_result is None:
        raise ModelFitError(
            f"No ARMA(p,q) order in range 0..{max_order} converged for this series."
        )
    return best_result


def _fit_arma_single(train: pd.Series, order: tuple[int, int]) -> ARMAResult:
    p, q = order
    try:
        model = ARIMA(train, order=(p, 0, q))
        fitted = model.fit()
    except Exception as exc:  # statsmodels raises various error types on non-convergence
        raise ModelFitError(f"ARMA({p},{q}) failed to fit: {exc}") from exc

    return ARMAResult(order=order, fitted_model=fitted, resid=fitted.resid)


# ---------------------------------------------------------------------------
# GARCH (two-step: ARMA mean, then GARCH(1,1) on ARMA residuals)
# ---------------------------------------------------------------------------

@dataclass
class GARCHResult:
    """Wraps the two-step ARMA-mean + GARCH(1,1)-variance fit."""

    arma: ARMAResult
    garch_fitted_model: object  # arch ARCHModelResult
    _resid_scale: float  # factor residuals were multiplied by before fitting GARCH

    def forecast(self, n_steps: int) -> dict[str, pd.Series]:
        """
        Forecast both the conditional mean (from the ARMA leg) and the
        conditional variance (from the GARCH leg) for the next n_steps.

        Returns
        -------
        dict with keys "mean" and "variance", each a pd.Series of length
        n_steps with a plain integer index. Both are on the original
        (unscaled) log-return scale.
        """
        mean_forecast = self.arma.forecast(n_steps)

        garch_fc = self.garch_fitted_model.forecast(horizon=n_steps, reindex=False)
        # GARCH was fit on residuals scaled by _resid_scale for numerical
        # stability, so variance (which scales quadratically) must be
        # divided by _resid_scale**2 to return to the original scale.
        variance_forecast = pd.Series(
            garch_fc.variance.values[-1, :] / (self._resid_scale**2), index=range(n_steps)
        )

        return {"mean": mean_forecast, "variance": variance_forecast}


def fit_garch(
    train: pd.Series,
    arma_order: tuple[int, int] | None = None,
    max_order: int = 5,
) -> GARCHResult:
    """
    Fit an ARMA-mean, GARCH(1,1)-variance model (proposal Section 5.2, model 3),
    using the two-step approach described in the module docstring.

    Parameters
    ----------
    train : pd.Series
        Training log-returns.
    arma_order : tuple[int, int], optional
        Fixed (p, q) for the mean equation. If None, selected by BIC
        (same as `fit_arma`).
    max_order : int
        Upper bound for the BIC search when `arma_order` is not given.

    Returns
    -------
    GARCHResult

    Raises
    ------
    ModelFitError
        If the ARMA mean step or the GARCH variance step fails to fit.
    """
    arma_result = fit_arma(train, order=arma_order, max_order=max_order)

    # arch_model expects residuals on a percentage-like scale to avoid
    # convergence issues with very small (log-return-scale) numbers.
    resid_scale = 100.0
    resid_scaled = arma_result.resid * resid_scale

    from arch import arch_model  # imported lazily to keep import cost out of fit_arma-only paths

    try:
        garch_spec = arch_model(resid_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        garch_fitted = garch_spec.fit(disp="off")
    except Exception as exc:
        raise ModelFitError(f"GARCH(1,1) failed to fit on ARMA residuals: {exc}") from exc

    return GARCHResult(arma=arma_result, garch_fitted_model=garch_fitted, _resid_scale=resid_scale)


# ---------------------------------------------------------------------------
# GARCH-in-mean (flagged as outstanding in week_02 lab log; implemented here)
# ---------------------------------------------------------------------------

@dataclass
class GARCHInMeanResult:
    """
    Result of an AR(1)-GARCH(1,1)-in-mean fit (Engle, Lilien & Robins, 1987):

        r_t      = c + phi * r_{t-1} + lam * sigma_t^2 + eps_t
        sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2

    `lam` is the price-of-risk coefficient: the proposal's GARCH-M test asks
    whether conditional variance itself helps predict next-day return, i.e.
    whether `lam` is significantly different from zero.

    Estimation note
    ----------------
    The conditional variance enters the mean equation contemporaneously, so
    the textbook estimator is full joint MLE over both equations at once.
    The `arch` package's mean specifications (Zero/Constant/AR/ARX/HAR/HARX)
    have no "plug the model's own conditional variance back into the mean"
    option, so joint MLE isn't available off the shelf here (same library
    limitation noted in the `fit_garch` docstring above for the ARMA+GARCH
    case). This implementation instead uses the standard backfitting
    approximation to joint MLE:

        1. Fit a zero-mean GARCH(1,1) to get an initial conditional
           variance series sigma_t^2.
        2. Estimate the mean equation r_t = c + phi*r_{t-1} + lam*sigma_t^2 + eps_t
           by OLS, using the GARCH-implied sigma_t^2 as a regressor.
        3. Refit GARCH(1,1) on the new residuals eps_t from step 2.
        4. Repeat steps 2-3 for `n_iter` rounds (default 3); coefficients
           typically stabilise within 2-3 iterations.

    This is an approximation, not exact joint MLE — flagged explicitly here
    and should be described as such in the dissertation's methodology
    section, not presented as if `arch_model`'s native ARX-GARCH fit
    produced it directly.
    """

    omega: float
    alpha: float
    beta: float
    const: float
    phi: float
    lam: float
    lam_tstat: float
    lam_pvalue: float
    n_iter: int
    converged: bool
    garch_fitted_model: object
    _resid_scale: float

    def forecast(self, n_steps: int = 1) -> dict[str, float]:
        """One-step-ahead mean and variance forecast (n_steps > 1 not supported,
        since lam*sigma_t^2 feedback into multi-step variance forecasts is not
        implemented here)."""
        if n_steps != 1:
            raise NotImplementedError(
                "GARCHInMeanResult.forecast only supports n_steps=1; the "
                "feedback loop between forecasted variance and the mean "
                "equation is not implemented for multi-step horizons."
            )
        garch_fc = self.garch_fitted_model.forecast(horizon=1, reindex=False)
        sigma2 = float(garch_fc.variance.values[-1, 0]) / (self._resid_scale**2)
        mean = self.const + self.lam * sigma2  # phi*r_{t-1} term added by caller, which holds r_{t-1}
        return {"mean": mean, "variance": sigma2}


def fit_garch_in_mean(train: pd.Series, n_iter: int = 3) -> GARCHInMeanResult:
    """
    Fit AR(1)-GARCH(1,1)-in-mean via iterative backfitting (see
    `GARCHInMeanResult` docstring for why this approximates, rather than
    replicates, joint MLE).

    Parameters
    ----------
    train : pd.Series
        Training log-returns.
    n_iter : int
        Number of backfitting rounds (mean-equation OLS <-> GARCH refit).

    Returns
    -------
    GARCHInMeanResult

    Raises
    ------
    ModelFitError
        If any GARCH refit step fails to converge.
    """
    import statsmodels.api as sm
    from arch import arch_model

    resid_scale = 100.0
    r = train.dropna().astype("float64") * resid_scale
    r_lag = r.shift(1)

    # Step 1: zero-mean GARCH(1,1) to initialise sigma_t^2
    try:
        garch_fitted = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    except Exception as exc:
        raise ModelFitError(f"Initial GARCH(1,1) failed to fit: {exc}") from exc

    converged = False
    const = phi = lam = lam_t = lam_p = 0.0

    for i in range(n_iter):
        sigma2 = garch_fitted.conditional_volatility**2  # on the resid_scale-d series

        X = pd.DataFrame({"r_lag": r_lag, "sigma2": sigma2})
        X = sm.add_constant(X)
        y = r
        common = X.dropna().index.intersection(y.dropna().index)
        if len(common) < 30:
            raise ModelFitError("GARCH-in-mean: too few overlapping observations after lagging/dropna.")

        ols_fit = sm.OLS(y.loc[common], X.loc[common]).fit()
        const, phi, lam = ols_fit.params["const"], ols_fit.params["r_lag"], ols_fit.params["sigma2"]
        lam_t, lam_p = ols_fit.tvalues["sigma2"], ols_fit.pvalues["sigma2"]

        new_resid = (y.loc[common] - ols_fit.predict(X.loc[common])).reindex(r.index)
        new_resid = new_resid.dropna()

        try:
            garch_fitted = arch_model(
                new_resid, mean="Zero", vol="GARCH", p=1, q=1, dist="normal"
            ).fit(disp="off")
        except Exception as exc:
            raise ModelFitError(f"GARCH-in-mean refit failed at iteration {i + 1}: {exc}") from exc

        converged = True

    omega = float(garch_fitted.params["omega"])
    alpha = float(garch_fitted.params["alpha[1]"])
    beta = float(garch_fitted.params["beta[1]"])

    return GARCHInMeanResult(
        omega=omega,
        alpha=alpha,
        beta=beta,
        const=float(const),
        phi=float(phi),
        lam=float(lam),
        lam_tstat=float(lam_t),
        lam_pvalue=float(lam_p),
        n_iter=n_iter,
        converged=converged,
        garch_fitted_model=garch_fitted,
        _resid_scale=resid_scale,
    )
