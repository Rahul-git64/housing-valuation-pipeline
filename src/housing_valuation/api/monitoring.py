"""Covariate and income-to-price affordability drift supervision via KS-test and PSI."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from housing_valuation.core.config import AppConfig


def compute_psi(reference_quantiles: list[float], current_values: np.ndarray) -> float:
    """Compute Population Stability Index (PSI) using reference decile bin edges."""
    if len(current_values) == 0 or len(reference_quantiles) < 3:
        return 0.0

    edges = np.unique(np.array(reference_quantiles, dtype=float))
    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf
    n_bins = len(edges) - 1

    expected_probs = np.full(n_bins, 1.0 / n_bins)
    counts, _ = np.histogram(current_values, bins=edges)
    actual_probs = counts / max(len(current_values), 1)

    eps = 1e-4
    expected_safe = np.clip(expected_probs, eps, 1.0)
    actual_safe = np.clip(actual_probs, eps, 1.0)

    psi_value = float(np.sum((actual_safe - expected_safe) * np.log(actual_safe / expected_safe)))
    return round(max(0.0, psi_value), 4)


class DriftSupervisor:
    """Maintains a sliding window of inference requests and evaluates feature + affordability drift."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.buffer: deque[dict[str, Any]] = deque(maxlen=config.monitoring.window_size)

    def record_request(self, payload: dict[str, Any], income_to_price_ratio: float) -> None:
        """Append a validated inference payload and its computed affordability ratio."""
        entry = dict(payload)
        entry["_income_to_price_ratio"] = float(income_to_price_ratio)
        self.buffer.append(entry)

    def evaluate(self, reference_stats: dict[str, Any]) -> dict[str, Any]:
        """Compare buffered inference payloads and affordability ratios against training baseline."""
        records = list(self.buffer)
        ref_size = int(reference_stats.get("n_reference", 0))

        if len(records) < 5:
            return {
                "status": "insufficient_data",
                "overall_drift_detected": False,
                "sample_size": len(records),
                "reference_size": ref_size,
                "income_to_price_ratio_drift": {
                    "ks_statistic": 0.0,
                    "ks_p_value": 1.0,
                    "psi": 0.0,
                    "mean_reference": round(
                        float(reference_stats.get("income_to_price_ratio", {}).get("mean", 0.0)),
                        3,
                    ),
                    "mean_current": 0.0,
                    "drift_detected": False,
                },
                "numerical_features": {},
                "categorical_features": {},
            }

        num_report: dict[str, Any] = {}
        cat_report: dict[str, Any] = {}
        any_drift = False

        for col in self.config.features.numerical_columns:
            col_ref = reference_stats.get("numerical", {}).get(col, {})
            ref_sample = np.array(col_ref.get("sample", []), dtype=float)
            cur_vals = np.array([float(r[col]) for r in records if col in r], dtype=float)

            if len(ref_sample) > 0 and len(cur_vals) > 0:
                ks_res = ks_2samp(ref_sample, cur_vals)
                ks_stat = round(float(ks_res.statistic), 4)
                ks_pval = round(float(ks_res.pvalue), 4)
            else:
                ks_stat, ks_pval = 0.0, 1.0

            psi_val = compute_psi(col_ref.get("quantiles", []), cur_vals)
            drifted = bool(
                ks_pval < self.config.monitoring.ks_alpha
                and psi_val >= self.config.monitoring.psi_threshold
            )
            if drifted:
                any_drift = True

            num_report[col] = {
                "ks_statistic": ks_stat,
                "ks_p_value": ks_pval,
                "psi": psi_val,
                "mean_reference": round(float(col_ref.get("mean", 0.0)), 3),
                "mean_current": round(float(np.mean(cur_vals)), 3),
                "drift_detected": drifted,
            }

        for col in self.config.features.categorical_columns:
            ref_dist: dict[str, float] = reference_stats.get("categorical", {}).get(col, {})
            cur_counts: dict[str, int] = {}
            for r in records:
                val = str(r.get(col, ""))
                cur_counts[val] = cur_counts.get(val, 0) + 1

            total = max(len(records), 1)
            all_cats = set(ref_dist.keys()) | set(cur_counts.keys())
            l1_div = 0.5 * sum(
                abs(cur_counts.get(cat, 0) / total - float(ref_dist.get(cat, 0.0)))
                for cat in all_cats
            )
            l1_rounded = round(float(l1_div), 4)
            drifted = bool(l1_rounded >= self.config.monitoring.categorical_l1_threshold)
            if drifted:
                any_drift = True

            cat_report[col] = {
                "l1_divergence": l1_rounded,
                "drift_detected": drifted,
            }

        # Evaluate income-to-price ratio affordability shift
        pti_ref = reference_stats.get("income_to_price_ratio", {})
        pti_ref_sample = np.array(pti_ref.get("sample", []), dtype=float)
        pti_cur = np.array(
            [float(r["_income_to_price_ratio"]) for r in records if "_income_to_price_ratio" in r],
            dtype=float,
        )
        if len(pti_ref_sample) > 0 and len(pti_cur) > 0:
            pti_ks = ks_2samp(pti_ref_sample, pti_cur)
            pti_ks_stat = round(float(pti_ks.statistic), 4)
            pti_ks_pval = round(float(pti_ks.pvalue), 4)
        else:
            pti_ks_stat, pti_ks_pval = 0.0, 1.0

        pti_psi = compute_psi(pti_ref.get("quantiles", []), pti_cur)
        pti_drifted = bool(
            pti_ks_pval < self.config.monitoring.ks_alpha
            and pti_psi >= self.config.monitoring.income_price_ratio_psi_threshold
        )
        if pti_drifted:
            any_drift = True

        pti_report = {
            "ks_statistic": pti_ks_stat,
            "ks_p_value": pti_ks_pval,
            "psi": pti_psi,
            "mean_reference": round(float(pti_ref.get("mean", 0.0)), 3),
            "mean_current": round(float(np.mean(pti_cur)) if len(pti_cur) else 0.0, 3),
            "drift_detected": pti_drifted,
        }

        return {
            "status": "drift_detected" if any_drift else "healthy",
            "overall_drift_detected": any_drift,
            "sample_size": len(records),
            "reference_size": ref_size,
            "income_to_price_ratio_drift": pti_report,
            "numerical_features": num_report,
            "categorical_features": cat_report,
        }
