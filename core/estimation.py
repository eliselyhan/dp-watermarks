"""Estimate watermark frequency and summarize errors."""
import math
import numpy as np


def estimate(favored_fraction, p0, epsilon, parallel=False):
    """Only the unlabeled mixture's favored fraction enters the estimator."""
    # Estimator: derive the watermarked favored probability, then invert
    # f = (1 - lambda) * p0 + lambda * p1 using only the observed fraction.
    p1 = (min(p0/math.exp(-epsilon), 1-math.exp(-epsilon)*(1-p0)) if parallel
          else p0/(p0+math.exp(-epsilon)*(1-p0)))
    value = (favored_fraction-p0)/(p1-p0)
    return np.clip(value, 0, 1)


# Summarize existing estimates against the true lambda; this does not
# construct pools or draw new response samples.
def statistics(values, lam):
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0, 1)
    low, high = np.quantile(values, [.05, .95])
    errors = 100*np.abs(values-lam)/lam
    error_low, error_high = np.quantile(errors, [.05, .95])
    return dict(count=len(values), mean=float(values.mean()), p05=float(low),
                p95=float(high), mape_p05=float(error_low), mape_p95=float(error_high), mape=float(np.mean(100*np.abs(values-lam)/lam)))
