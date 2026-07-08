"""Incremental statistics for chunked DTA data dictionary generation."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from src.weighted_freq_key import normalize_category_value


def _combine_welford(
    n_a: int, mean_a: float, m2_a: float, n_b: int, mean_b: float, m2_b: float
) -> tuple[int, float, float]:
    if n_b == 0:
        return n_a, mean_a, m2_a
    if n_a == 0:
        return n_b, mean_b, m2_b
    n = n_a + n_b
    delta = mean_b - mean_a
    mean = mean_a + delta * n_b / n
    m2 = m2_a + m2_b + delta * delta * n_a * n_b / n
    return n, mean, m2


class _ColumnStats:
    __slots__ = (
        "valid_count",
        "invalid_count",
        "is_numeric",
        "min_value",
        "max_value",
        "n",
        "mean",
        "m2",
        "value_counts",
    )

    def __init__(self) -> None:
        self.valid_count = 0
        self.invalid_count = 0
        self.is_numeric: bool | None = None
        self.min_value: float | None = None
        self.max_value: float | None = None
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.value_counts: dict[object, int] = defaultdict(int)

    def update(self, series: pd.Series, user_missings: list | None = None) -> None:
        user_missings = user_missings or []
        working = series.replace(user_missings, np.nan)
        invalid = int(working.isna().sum())
        valid = working.dropna()
        self.invalid_count += invalid
        self.valid_count += int(valid.count())

        if len(valid) == 0:
            return

        counts = valid.value_counts()
        for key, freq in counts.items():
            canonical = normalize_category_value(key)
            self.value_counts[canonical] += int(freq)

        if self.is_numeric is None:
            self.is_numeric = pd.api.types.is_numeric_dtype(valid)

        if not self.is_numeric:
            return

        values = valid.astype(float)
        chunk_n = len(values)
        chunk_mean = float(values.mean())
        chunk_m2 = float(((values - chunk_mean) ** 2).sum())
        chunk_min = float(values.min())
        chunk_max = float(values.max())
        self.n, self.mean, self.m2 = _combine_welford(
            self.n, self.mean, self.m2, chunk_n, chunk_mean, chunk_m2
        )
        self.min_value = (
            chunk_min if self.min_value is None else min(self.min_value, chunk_min)
        )
        self.max_value = (
            chunk_max if self.max_value is None else max(self.max_value, chunk_max)
        )

    def stddev(self) -> float | None:
        if not self.is_numeric or self.n < 2:
            return None
        return float(np.sqrt(self.m2 / (self.n - 1)))


class _WeightedStats:
    __slots__ = ("sum_weights", "sum_wx", "sum_wx2", "freq")

    def __init__(self) -> None:
        self.sum_weights = 0.0
        self.sum_wx = 0.0
        self.sum_wx2 = 0.0
        self.freq: dict[object, float] = defaultdict(float)

    def update(
        self,
        values: pd.Series,
        weights: pd.Series,
    ) -> None:
        w = weights.astype(float)
        x = values.astype(float)
        self.sum_weights += float(w.sum())
        self.sum_wx += float((w * x).sum())
        self.sum_wx2 += float((w * x * x).sum())
        grouped = pd.DataFrame({"x": x, "w": w}).groupby("x")["w"].sum()
        for key, weight_sum in grouped.items():
            self.freq[key] += float(weight_sum)

    def mean(self) -> float | None:
        if self.sum_weights == 0:
            return None
        return self.sum_wx / self.sum_weights

    def stddev(self) -> float | None:
        if self.sum_weights <= 1:
            return None
        variance = (self.sum_wx2 - (self.sum_wx**2) / self.sum_weights) / (
            self.sum_weights - 1
        )
        if variance < 0:
            variance = 0.0
        return float(np.sqrt(variance))


class ChunkedDictionaryStats:
    """Accumulate per-column stats across DTA chunks."""

    def __init__(self, column_names: list[str]) -> None:
        self.column_names = column_names
        self.columns: dict[str, _ColumnStats] = {
            name: _ColumnStats() for name in column_names
        }
        self.weights: dict[str, _WeightedStats] = {}

    def update_chunk(
        self,
        df: pd.DataFrame,
        missings: dict,
        weight_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        for name in self.column_names:
            if name not in df.columns:
                continue
            user_missings = missings.get(name, []) if missings else []
            if user_missings and not isinstance(user_missings, list):
                user_missings = list(user_missings)
            self.columns[name].update(df[name], user_missings=user_missings)

        for field, weight_field in weight_pairs or []:
            if field not in df.columns or weight_field not in df.columns:
                continue
            key = field
            if key not in self.weights:
                self.weights[key] = _WeightedStats()
            working = df[[field, weight_field]].copy()
            field_missings = missings.get(field, []) if missings else []
            weight_missings = missings.get(weight_field, []) if missings else []
            if field_missings:
                working[field] = working[field].replace(field_missings, np.nan)
            if weight_missings:
                working[weight_field] = working[weight_field].replace(
                    weight_missings, np.nan
                )
            working.dropna(inplace=True)
            if working.empty:
                continue
            self.weights[key].update(working[field], working[weight_field])

    def column_stats(self, name: str) -> _ColumnStats:
        return self.columns[name]

    def weighted_stats(self, field: str) -> _WeightedStats | None:
        return self.weights.get(field)
