"""Whether to compute a frequency distribution nobody asked for.

This decides one thing only: on a first read, before the editor has an opinion,
should we spend a frequency list on this variable? It does **not** decide what
the variable is. Measurement level comes from the file (SPSS states it; Stata's
only statement is that a value label is attached) and is reported separately.

The editor owns the real decision, per variable, through `sum_stats_options.freq`,
which arrives here as `categorical`. That list always wins. But it is empty on a
first upload — the variables do not exist yet — so something has to choose a
starting point, and "every labelled variable" is the wrong answer: a continuous
variable that labels `99 = Refused` would produce one entry per distinct value.

So: compute when the number of distinct values does not exceed the number of
value labels; defer when it does (e.g. 1000 distinct, 10 labels). No key
matching — only the two counts. Either way the caller reports
`distinct_value_count` and `labelled_value_count`, so the editor can show what
was skipped and offer the choice instead of guessing again.

The same applies to the configured limits. A list is never truncated to fit one —
percentages divide by valid cases, so a partial distribution renders as a real one
that quietly stops adding up, and DDI cannot express "top N". Over a limit, the list
is skipped and the reason reported.

Every decision here is about one variable and its own values. Nothing depends on
the rest of the file, so the same variable always gets the same answer.
"""

import logging

from src.weighted_freq_key import normalize_category_value

logger = logging.getLogger(__name__)


def _canonical_label_map(categories):
    """
    Value-label dict keyed by normalize_category_value (string keys), so int 1
    and str "1" join when attaching labels to frequency rows. Padded codes stay
    distinct ("01" ≠ "1").
    """
    out = {}
    for key, label in (categories or {}).items():
        canon = normalize_category_value(key)
        if canon not in out:
            out[canon] = label
    return out


def label_for(value, categories, is_numeric=False):
    """Look up a value's label using the same string keys frequency counts use."""
    if not categories:
        return None
    if value in categories:
        return categories[value]
    canon = normalize_category_value(value)
    if canon in categories:
        return categories[canon]
    mapped = _canonical_label_map(categories)
    if canon in mapped:
        return mapped[canon]
    return None


def frequency_skip_reason(
    categories,
    value_counts,
    is_numeric=False,
    is_user_categorical=False,
    auto_limit=0,
    max_limit=0,
):
    """
    Why this variable should not get a frequency list, or None to compute one.

    Auto rule: distinct values must not exceed the number of value labels.
    Limits of 0 are unlimited. A list is never truncated to fit a limit.
    """
    distinct = len(value_counts)
    label_count = len(categories or {})

    if is_user_categorical:
        # Max limit still applies to an explicit request — protects the whole import.
        if max_limit and distinct > max_limit:
            return "over_max_limit"
        return None

    if not categories:
        return "not_labelled"

    if max_limit and distinct > max_limit:
        return "over_max_limit"

    if auto_limit and distinct > auto_limit:
        return "over_auto_limit"

    if not value_counts:
        # No data to judge by (metadata-only read): labels are the only signal.
        return None

    if distinct > label_count:
        return "labels_did_not_cover"

    return None


def plan_frequencies(
    categories,
    values,
    is_numeric=False,
    is_user_categorical=False,
    auto_limit=0,
    max_limit=0,
):
    """
    What we know about a variable's values, and whether to spend a list on them.

    `values` is any iterable of the distinct values observed, already canonicalised.
    `labelled_value_count` is the size of the file's value-label map (not a match count).
    """
    values = values if isinstance(values, (set, frozenset, dict)) else set(values)
    categories = categories or {}
    return {
        "distinct_value_count": len(values),
        "labelled_value_count": len(categories),
        "skip_reason": frequency_skip_reason(
            categories,
            values,
            is_numeric=is_numeric,
            is_user_categorical=is_user_categorical,
            auto_limit=auto_limit,
            max_limit=max_limit,
        ),
    }


def plan_frequencies_for_file(
    params, column_names, values_for, is_numeric_for, value_labels
):
    """
    Plan every variable in a file in one pass.

    Each variable is judged on its own values only — nothing here depends on what the rest
    of the file contains, so a result never changes because of the company it keeps. Shared
    by the chunked, in-memory and CSV paths so the rule cannot drift between them.

    Returns a plan per variable name; see plan_frequencies().
    """
    categorical_list = params.categorical or []
    plans = {}
    for name in column_names:
        plans[name] = plan_frequencies(
            value_labels.get(name, {}),
            values_for(name),
            is_numeric_for(name),
            name in categorical_list,
            auto_limit=params.frequency_auto_limit,
            max_limit=params.frequency_max_limit,
        )
        if plans[name]["skip_reason"] is not None:
            logger.debug(
                f"Variable {name} skipped frequencies: "
                f"{plans[name]['skip_reason']} "
                f"({plans[name]['distinct_value_count']} distinct values)"
            )

    return plans
