"""Which columns actually drive the target — and which only look like they do?

Shared by the regression and classification recipes. Both already screen out
leaks, identifiers, and over-wide categoricals before probing; this module takes
the feature groups they produced rather than selecting its own, so the columns a
tree ranks are provably the columns the linear probe trained on. A forest fitted
on the leak would rank the leak first and bury the finding that matters.

Three deliberate choices.

**Two importance measures, and their disagreement is a finding.** Impurity
importance (MDI) is free — it falls out of the fitted forest — and it is
systematically inflated for high-cardinality and continuous columns, because a
column with many distinct values offers many split points and will reduce
impurity somewhere by luck. Permutation importance on held-out rows has no such
bias. Reporting only MDI would rank a near-unique reference code above a real
driver; reporting only permutation would hide *why* that happens. So both are
measured, and a column that ranks high by impurity while losing to noise by
permutation is reported as exactly what it is.

**A tree always produces a ranking, even from pure noise.** Two gates therefore
stand in front of every number here. The model must first beat a dummy baseline
on held-out rows at all — importance from a model that cannot predict is not
weak evidence, it is no evidence, and the section does not render. Then, for a
model that does predict, the floor for "this column beat noise" is *measured*
rather than assumed: three sentinel columns of manufactured noise are given to
the model alongside the real features, and the best importance any of them
achieves is the floor. One is gaussian, one uniform, and one a shuffled copy of
the widest-cardinality real feature — that third one is the calibration that
matters, because it has a real column's marginal distribution and cardinality
and none of its information.

**Permutation splits credit between correlated columns.** Two features that
correlate at 0.9999 each look useless when permuted, because the model reads the
survivor whenever one is shuffled. Dropping both would lose the signal entirely,
so a column with a redundant partner never reaches the drop list — it is
reported, with the reason.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prism_eda.catalog.models import TableCatalog
from prism_eda.config import AnalysisConfig, AnalysisMode
from prism_eda.evidence.models import (
    OBSERVATION,
    Evidence,
    EvidenceScope,
    Finding,
)
from prism_eda.results import AnalysisWarning, SamplingRecord
from prism_eda.transformations.models import TransformationStep

Task = Literal["regression", "classification"]

#: The one evidence kind this module produces. Deliberately not prefixed by
#: recipe: the section, the chart, and the findings are identical for both
#: tasks, and ``value["task"]`` carries the difference.
EVIDENCE_KIND = "feature_importance"

#: Bars drawn on the chart. Everything past this still travels in the evidence
#: and renders in the report's expandable table — a cap may not drop a column
#: silently.
MAX_CHART_FEATURES = 15

#: Permutation repeats per feature, by compute mode. Each repeat is one extra
#: scoring pass per feature, so this is the term that dominates the stage's cost.
_REPEATS = {
    AnalysisMode.QUICK: 3,
    AnalysisMode.STANDARD: 5,
    AnalysisMode.DEEP: 10,
}

#: Trees in the forest, by mode. More trees stabilize the importances; past a
#: few hundred the ranking stops moving.
_TREES = {
    AnalysisMode.QUICK: 100,
    AnalysisMode.STANDARD: 200,
    AnalysisMode.DEEP: 300,
}

#: Rows entering this stage, capped independently of the recipe's own budget the
#: way the influence design already is. Permutation cost is linear in rows *and*
#: multiplied by features and repeats, so it needs its own ceiling.
_ROW_CAPS = {
    AnalysisMode.QUICK: 25_000,
    AnalysisMode.STANDARD: 50_000,
    AnalysisMode.DEEP: 100_000,
}

#: Share of rows held out for scoring and permutation. The forest never sees
#: them, so neither the skill gate nor the noise floor is an in-sample number.
_HOLDOUT_FRACTION = 0.25

#: Below this many usable rows a 75/25 split leaves a holdout too small for a
#: permutation estimate to mean anything.
_MIN_IMPORTANCE_ROWS = 80

#: Rounding applied before banking, so a difference in floating-point reduction
#: order cannot mint a new evidence ID for an identical ranking. Same reasoning
#: as the clustering metrics.
_STABLE_DIGITS = 12

_SENTINEL_PREFIX = "__prism_sentinel"
_SENTINEL_NORMAL = f"{_SENTINEL_PREFIX}_normal__"
_SENTINEL_UNIFORM = f"{_SENTINEL_PREFIX}_uniform__"
_SENTINEL_SHUFFLED = f"{_SENTINEL_PREFIX}_shuffled__"

#: Gate 1. The forest must clear the dummy baseline by this much on held-out
#: rows before any importance is reported at all.
_MIN_SKILL_MARGIN = 0.02

#: Two columns at or above this absolute correlation carry one relationship.
#: Permutation splits their credit, so neither may be recommended for dropping.
#: This mirrors the regression recipe's own redundancy screen, and is kept here
#: so the module stands alone for classification too, which has no such screen.
_REDUNDANCY_CORRELATION = 0.95

# Thresholds for the three promoted findings. Each is set where a clean,
# well-specified dataset produces nothing: a detector that never stays quiet is
# not a detector.
#
# Cardinality bias: a column must rank near the top by impurity, hold a real
# share of it, and still lose to manufactured noise when permuted. A genuinely
# important column clears the floor, so this cannot fire on honest signal.
#
# The third guard is the sentinels' *impurity*, not just their permutation
# score. The sentinels are pure noise with a real column's shape, so whatever
# impurity they collect is what cardinality alone buys in this fit. A column
# whose impurity sits inside that range is being ranked by its number of split
# points; one whose impurity is many times higher is being ranked by something
# the fit can actually see, even when permutation cannot confirm it out of
# sample, and this detector must leave that column alone.
#
# What the guards cannot do is separate a decoy from a real-but-masked feature:
# on a target driven by a step function, a genuine weak driver and a random
# 50-level code land at 1.6x and 1.9x the sentinels' impurity. So the finding
# reports the measurement — high in sample, beaten by noise out of sample — and
# names masking as a possible cause rather than asserting cardinality.
_BIAS_MAX_RANK = 3
_BIAS_MIN_IMPURITY_SHARE = 0.10
_BIAS_SENTINEL_IMPURITY_FACTOR = 2.5
_MAX_BIAS_FINDINGS = 3

# Tree-beats-linear: on linear data the two scores match within noise, so the
# gap must be large in absolute terms, large relative to what the linear probe
# managed, and coming from a forest that is itself worth listening to.
_TREE_GAIN_ABSOLUTE = 0.10
_TREE_GAIN_RELATIVE = 0.25
_TREE_GAIN_MIN_SCORE = 0.20

# Dominance exists to catch a *soft leak*: a column that is a strong proxy for
# the target but sits under the 98%-explained-variance bar the leakage screen
# uses. Share of importance turned out to be the wrong statistic for that. On
# ``y = 2*x1 + 3*x2 + noise`` — a textbook clean regression — the coefficients
# and spreads give x1 four times x2's importance, and any share-based rule near
# that value either fires on honest data or misses real leaks. Which is what the
# suite's clean-data test proved.
#
# So the claim is measured instead of inferred. The same forest is refit on the
# top feature *alone*, on the same split, and the question becomes the one the
# finding actually makes: does that column by itself recover essentially all of
# what the full model achieved? On the clean frame it recovers about 85% and the
# detector stays quiet; a genuine proxy recovers nearly all of it.
_DOMINANCE_SOLO_RATIO = 0.90
_DOMINANCE_MIN_SCORE = 0.80
_DOMINANCE_MIN_FEATURES = 2


def _stable(value: float | None) -> float | None:
    """Round a metric to a precision that survives reduction order."""
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number):
        return number
    return float(f"{number:.{_STABLE_DIGITS}g}")


def _tree_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Impute and one-hot, with no scaling — a tree is scale-invariant.

    No ``drop="first"`` either. Dropping a level is a collinearity remedy for a
    linear model; for a tree it just hides one category from every split.
    """
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _linear_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """The scaled, level-dropped encoding a linear comparator needs."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop="first",
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _widest_feature(
    frame: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]
) -> str | None:
    """The admitted feature with the most distinct values.

    Its shuffled copy is the sentinel that matters: same marginal distribution,
    same cardinality, zero information. Whatever importance the model assigns it
    is the importance cardinality alone can buy.
    """
    candidates = numeric_features + categorical_features
    if not candidates:
        return None
    counts = {name: int(frame[name].nunique(dropna=True)) for name in candidates}
    # Sorted by name as the tiebreak so the choice is reproducible.
    return max(sorted(counts), key=lambda name: counts[name])


def _redundant_features(frame: pd.DataFrame, numeric_features: list[str]) -> set[str]:
    """Numeric columns that have a near-interchangeable partner."""
    if len(numeric_features) < 2:
        return set()
    design = frame[numeric_features].apply(pd.to_numeric, errors="coerce")
    design = design.loc[:, design.nunique(dropna=True) > 1]
    if design.shape[1] < 2:
        return set()
    correlations = design.corr(method="pearson").abs()
    partnered: set[str] = set()
    columns = list(correlations.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            value = correlations.loc[left, right]
            if pd.notna(value) and float(value) >= _REDUNDANCY_CORRELATION:
                partnered.add(str(left))
                partnered.add(str(right))
    return partnered


def _impurity_by_source_column(
    fitted: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, float]:
    """Sum the forest's per-encoded-column impurity back onto source columns.

    One categorical column becomes one dummy per level, and each dummy carries
    its own slice of impurity reduction. Reporting those separately would name
    columns the user does not have, so they are summed. The mapping is derived
    from the encoder's own ``categories_`` rather than by parsing generated
    feature names, which would be ambiguous whenever one column name is a
    prefix of another's.
    """
    importances = np.asarray(fitted.named_steps["model"].feature_importances_)
    mapped: dict[str, float] = {}
    offset = 0
    if numeric_features:
        for position, name in enumerate(numeric_features):
            mapped[name] = float(importances[offset + position])
        offset += len(numeric_features)
    if categorical_features:
        encoder = (
            fitted.named_steps["preprocess"]
            .named_transformers_["categorical"]
            .named_steps["encoder"]
        )
        for name, categories in zip(
            categorical_features, encoder.categories_, strict=True
        ):
            width = len(categories)
            mapped[name] = float(importances[offset : offset + width].sum())
            offset += width
    return mapped


def _sample(
    frame: pd.DataFrame,
    *,
    table: str,
    config: AnalysisConfig,
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> pd.DataFrame:
    """Cut to this stage's own row cap, recording that it happened."""
    cap = _ROW_CAPS[AnalysisMode(config.mode)]
    if config.sampling == "disabled" or len(frame) <= cap:
        return frame
    sampled = frame.sample(n=cap, random_state=config.random_seed).sort_index()
    warnings.append(
        AnalysisWarning(
            code="sampled_feature_importance",
            message=(
                f"Feature importance for {table} was measured on a deterministic "
                f"{cap:,}-row sample of {len(frame):,} rows."
            ),
            table=table,
        )
    )
    sampling.append(
        SamplingRecord(
            operation="feature_importance",
            source_rows=len(frame),
            sampled_rows=cap,
            strategy="deterministic_pandas_sample",
            seed=config.random_seed,
            reason="permutation_cost_scales_with_rows_features_and_repeats",
            limitations=(
                "Importance describes the sampled rows. A driver that only acts "
                "in a rare region of the data may be under-weighted.",
            ),
        )
    )
    return sampled


def _split(
    work: pd.DataFrame,
    y: pd.Series,
    *,
    task: Task,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | None:
    """A deterministic hold-out split, stratified where that is possible."""
    stratify: pd.Series | None = None
    if task == "classification":
        counts = y.value_counts(dropna=False)
        # Stratifying needs at least two rows of every class; below that the
        # split is still valid, it just cannot be balanced.
        if int(counts.min()) >= 2 and len(counts) >= 2:
            stratify = y
    try:
        train_x, test_x, train_y, test_y = train_test_split(
            work,
            y,
            test_size=_HOLDOUT_FRACTION,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:  # pragma: no cover - degenerate class structure
        return None
    if len(test_x) < 10 or len(train_x) < 20:
        return None
    if task == "classification" and train_y.nunique() < 2:
        return None
    return train_x, test_x, train_y, test_y


def _score(task: Task, actual: pd.Series, predicted: Any) -> float:
    if task == "regression":
        return float(r2_score(actual, predicted))
    return float(balanced_accuracy_score(actual, predicted))


def importance_evidence(
    frame: pd.DataFrame,
    table: TableCatalog,
    target: str,
    target_series: pd.Series,
    *,
    task: Task,
    config: AnalysisConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    warnings: list[AnalysisWarning],
    sampling: list[SamplingRecord],
) -> Evidence | None:
    """Rank the recipe's screened features by a forest, or explain the silence.

    Returns ``None`` — and records a warning — whenever the forest fails to beat
    a dummy baseline on held-out rows. That is the first gate: a ranking read
    off a model with no skill is an ordering of noise, and printing it would be
    worse than printing nothing.
    """
    features = numeric_features + categorical_features
    if not features:
        return None

    usable = frame.loc[target_series.notna()]
    if len(usable) < _MIN_IMPORTANCE_ROWS:
        return None
    if target_series.loc[usable.index].nunique() < 2:
        return None
    usable = _sample(
        usable,
        table=table.name,
        config=config,
        warnings=warnings,
        sampling=sampling,
    )
    y = target_series.loc[usable.index]

    # Sentinels are added to a private copy. The caller's DataFrame is never
    # touched, and neither is the recipe's own working frame.
    work = usable[features].copy()
    generator = np.random.default_rng(config.random_seed)
    sentinel_numeric = [_SENTINEL_NORMAL, _SENTINEL_UNIFORM]
    work[_SENTINEL_NORMAL] = generator.standard_normal(len(work))
    work[_SENTINEL_UNIFORM] = generator.uniform(size=len(work))
    sentinels: list[dict[str, Any]] = [
        {"name": _SENTINEL_NORMAL, "kind": "gaussian_noise", "source": None},
        {"name": _SENTINEL_UNIFORM, "kind": "uniform_noise", "source": None},
    ]
    sentinel_categorical: list[str] = []
    widest = _widest_feature(usable, numeric_features, categorical_features)
    if widest is not None:
        shuffled = usable[widest].to_numpy(copy=True)
        generator.shuffle(shuffled)
        work[_SENTINEL_SHUFFLED] = shuffled
        if widest in categorical_features:
            sentinel_categorical.append(_SENTINEL_SHUFFLED)
        else:
            sentinel_numeric.append(_SENTINEL_SHUFFLED)
        sentinels.append(
            {
                "name": _SENTINEL_SHUFFLED,
                "kind": "shuffled_copy",
                "source": widest,
            }
        )

    fit_numeric = numeric_features + sentinel_numeric
    fit_categorical = categorical_features + sentinel_categorical

    split = _split(work, y, task=task, seed=config.random_seed)
    if split is None:
        return None
    train_x, test_x, train_y, test_y = split

    trees = _TREES[AnalysisMode(config.mode)]
    if task == "regression":
        model: Any = RandomForestRegressor(
            n_estimators=trees,
            random_state=config.random_seed,
            min_samples_leaf=2,
        )
        baseline: Any = DummyRegressor(strategy="median")
        linear: Any = Ridge(alpha=1.0)
        linear_name = "ridge"
        metric = "r2"
    else:
        model = RandomForestClassifier(
            n_estimators=trees,
            random_state=config.random_seed,
            min_samples_leaf=2,
            class_weight="balanced",
        )
        baseline = DummyClassifier(strategy="most_frequent")
        linear = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=0,
        )
        linear_name = "logistic_regression"
        metric = "balanced_accuracy"

    forest = Pipeline(
        [
            ("preprocess", _tree_preprocessor(fit_numeric, fit_categorical)),
            ("model", model),
        ]
    )
    try:
        forest.fit(train_x, train_y)
        model_score = _score(task, test_y, forest.predict(test_x))
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
        return None

    baseline.fit(train_x, train_y)
    baseline_score = _score(task, test_y, baseline.predict(test_x))

    # Gate 1. Importance read off a model that cannot predict is not weak
    # evidence; it is an ordering of noise. Say so and produce nothing.
    if model_score - baseline_score < _MIN_SKILL_MARGIN or (
        task == "regression" and model_score <= 0.0
    ):
        warnings.append(
            AnalysisWarning(
                code="feature_importance_no_signal",
                message=(
                    f"No tree-based model found signal for {table.name}.{target} "
                    f"({metric} {model_score:.3f} against a baseline of "
                    f"{baseline_score:.3f}), so no feature importance is reported. "
                    "Importance from a model that cannot predict would be an "
                    "ordering of noise."
                ),
                table=table.name,
                column=target,
            )
        )
        return None

    # The linear comparator is refit on this very split rather than reusing the
    # recipe's cross-validated probe, so "the tree does better" is a comparison
    # of two numbers measured the same way on the same rows. Sentinels are
    # excluded from it: they exist to calibrate the forest's noise floor, and
    # handing them to the comparator would only add noise to its score.
    linear_score: float | None = None
    try:
        comparator = Pipeline(
            [
                (
                    "preprocess",
                    _linear_preprocessor(numeric_features, categorical_features),
                ),
                ("model", linear),
            ]
        )
        comparator.fit(train_x[features], train_y)
        linear_score = _score(task, test_y, comparator.predict(test_x[features]))
    except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
        linear_score = None

    repeats = _REPEATS[AnalysisMode(config.mode)]
    permutation = permutation_importance(
        forest,
        test_x,
        test_y,
        scoring=metric,
        n_repeats=repeats,
        random_state=config.random_seed,
    )
    permuted = {
        str(name): (
            float(permutation.importances_mean[index]),
            float(permutation.importances_std[index]),
        )
        for index, name in enumerate(work.columns)
    }
    impurity = _impurity_by_source_column(forest, fit_numeric, fit_categorical)

    # The floor is the best a manufactured noise column managed *including its
    # own run-to-run wobble*. Taking the bare mean of three draws makes the
    # floor a noisy statistic in its own right, and a real noise column then
    # clears it about as often as not: on a clean frame a random three-level
    # categorical scored 0.00056 against a floor of 0.00039 and was reported as
    # carrying signal. Adding each sentinel's spread fixes the estimate at the
    # source rather than by raising an arbitrary margin.
    sentinel_names = [str(item["name"]) for item in sentinels]
    noise_floor = max(permuted[name][0] + permuted[name][1] for name in sentinel_names)
    # A negative floor would mark every real feature as clearing it, which is
    # not a claim the sentinels support. Zero is the weakest honest floor.
    noise_floor = max(0.0, noise_floor)
    for item in sentinels:
        name = str(item["name"])
        item["permutation_importance"] = _stable(permuted[name][0])
        item["permutation_std"] = _stable(permuted[name][1])
        item["impurity_importance"] = _stable(impurity.get(name, 0.0))

    # Whatever impurity a pure-noise column collected is what cardinality alone
    # buys in this fit, and it is the reference the bias flag is judged against.
    sentinel_impurity_ceiling = max(
        float(item["impurity_importance"] or 0.0) for item in sentinels
    )

    redundant = _redundant_features(usable, numeric_features)

    impurity_total = sum(max(0.0, impurity.get(name, 0.0)) for name in features)
    permutation_total = sum(max(0.0, permuted[name][0]) for name in features)
    by_impurity = sorted(features, key=lambda name: (-impurity.get(name, 0.0), name))
    impurity_rank = {name: index + 1 for index, name in enumerate(by_impurity)}

    rows: list[dict[str, Any]] = []
    for name in features:
        mean, deviation = permuted[name]
        mdi = impurity.get(name, 0.0)
        # Clearing the floor is judged on the feature's own worst repeat, not
        # its average one. A column whose spread is wider than its mean has not
        # beaten noise; it has landed above it once.
        below = (mean - deviation) <= noise_floor
        rows.append(
            {
                "feature": name,
                "permutation_importance": _stable(mean),
                "permutation_std": _stable(deviation),
                "impurity_importance": _stable(mdi),
                "impurity_rank": impurity_rank[name],
                "impurity_share": _stable(
                    max(0.0, mdi) / impurity_total if impurity_total > 0 else 0.0
                ),
                "permutation_share": _stable(
                    max(0.0, mean) / permutation_total if permutation_total > 0 else 0.0
                ),
                "below_noise_floor": bool(below),
                "cardinality_biased": bool(
                    below
                    and impurity_rank[name] <= _BIAS_MAX_RANK
                    and impurity_total > 0
                    and mdi / impurity_total >= _BIAS_MIN_IMPURITY_SHARE
                    and mdi
                    <= _BIAS_SENTINEL_IMPURITY_FACTOR * sentinel_impurity_ceiling
                ),
                "has_redundant_partner": name in redundant,
                "kind": "categorical" if name in categorical_features else "numeric",
            }
        )
    rows.sort(
        key=lambda row: (
            -(row["permutation_importance"] or 0.0),
            -(row["impurity_importance"] or 0.0),
            row["feature"],
        )
    )
    for position, row in enumerate(rows, start=1):
        row["permutation_rank"] = position

    # How much of the model is just its best column? Refitting the same forest
    # on that one feature answers it directly, where a share of importance only
    # approximates it — and approximates it badly whenever two honest features
    # have unequal coefficients. One extra fit on a single column is cheap.
    solo_score: float | None = None
    solo_reliance: float | None = None
    lift = model_score - baseline_score
    if rows and lift > 0:
        leader = str(rows[0]["feature"])
        try:
            solo = Pipeline(
                [
                    (
                        "preprocess",
                        _tree_preprocessor(
                            [leader] if leader in numeric_features else [],
                            [leader] if leader in categorical_features else [],
                        ),
                    ),
                    ("model", clone(model)),
                ]
            )
            solo.fit(train_x[[leader]], train_y)
            solo_score = _score(task, test_y, solo.predict(test_x[[leader]]))
            solo_reliance = (solo_score - baseline_score) / lift
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - degenerate
            solo_score = None
            solo_reliance = None

    tree_advantage = (
        _stable(model_score - linear_score) if linear_score is not None else None
    )
    return Evidence.create(
        kind=EVIDENCE_KIND,
        scope=EvidenceScope(table=table.name, columns=tuple(features + [target])),
        value={
            "task": task,
            "target": target,
            "model": (
                "random_forest_regressor"
                if task == "regression"
                else "random_forest_classifier"
            ),
            "metric": metric,
            "tree_count": trees,
            "model_score": _stable(model_score),
            "baseline_score": _stable(baseline_score),
            "baseline_model": (
                "median_baseline" if task == "regression" else "majority_baseline"
            ),
            "linear_model": linear_name if linear_score is not None else None,
            "linear_score": _stable(linear_score),
            "tree_advantage": tree_advantage,
            "top_feature_solo_score": _stable(solo_score),
            "top_feature_solo_reliance": _stable(solo_reliance),
            "noise_floor": _stable(noise_floor),
            "sentinel_impurity_ceiling": _stable(sentinel_impurity_ceiling),
            "sentinels": sentinels,
            "features": rows,
            "feature_count": len(rows),
            "chart_feature_count": min(MAX_CHART_FEATURES, len(rows)),
            "below_noise_floor_count": sum(
                1 for row in rows if row["below_noise_floor"]
            ),
            "train_row_count": int(len(train_x)),
            "holdout_row_count": int(len(test_x)),
            "permutation_repeats": repeats,
        },
        method="sentinel_calibrated_forest_importance_v1",
        description=(
            f"Tree-based feature importance for {table.name}.{target}, "
            "calibrated against sentinel noise columns."
        ),
        confidence=0.76,
        assumptions=(
            "Importance describes this forest on these rows. It is a diagnostic "
            "fit, not a production model, and a different model class can rank "
            "features differently.",
            "Permutation importance is measured on held-out rows the forest "
            "never saw; impurity importance is in-sample and inflated for "
            "high-cardinality columns.",
            "The noise floor is the best score any of three manufactured noise "
            "columns achieved in the same fit, so it is measured rather than "
            "assumed.",
            "Permutation splits credit between correlated features, so two "
            "interchangeable columns can both score low while the information "
            "they share is genuinely useful.",
        ),
    )


def importance_findings_and_steps(
    item: Evidence,
) -> tuple[list[Finding], list[TransformationStep]]:
    """Promote only what a reader must act on; the chart carries the rest.

    Three findings, each guarded so that clean, well-specified data produces
    none of them. Note what is deliberately *not* here: "these columns scored
    below the noise floor" is a suggestion for the next iteration, not a defect,
    so it reaches the reader as a transformation-plan step rather than as a
    finding.
    """
    findings: list[Finding] = []
    steps: list[TransformationStep] = []
    value = item.value
    table = item.scope.table or ""
    target = value["target"]
    metric = "R²" if value["metric"] == "r2" else "balanced accuracy"
    rows: list[dict[str, Any]] = value["features"]

    leader = rows[0]["feature"] if rows else ""
    biased = [row for row in rows if row["cardinality_biased"]][:_MAX_BIAS_FINDINGS]
    for row in biased:
        findings.append(
            Finding.create(
                title=f"{row['feature']} ranks high in sample but not out of it",
                summary=(
                    f"{row['feature']} is #{row['impurity_rank']} of "
                    f"{value['feature_count']} by impurity importance, holding "
                    f"{row['impurity_share']:.1%} of the total — no more than the "
                    "manufactured noise columns given to the same forest "
                    "collected. Permuted on held-out rows it scores "
                    f"{row['permutation_importance']:.4f}, at or below the "
                    f"{value['noise_floor']:.4f} those noise columns reached."
                ),
                severity="medium",
                confidence=0.8,
                evidence_ids=(item.id,),
                recommendation=(
                    "Read the permutation figure, not the impurity rank: impurity "
                    "is in-sample and rewards columns with many split points, so "
                    "a wide column climbs it for free. This column carried no "
                    "signal this model could use. If you expected it to matter, "
                    + (
                        f"it may be masked by {leader}, "
                        if leader and leader != row["feature"]
                        else ""
                    )
                    + "or it may need bucketing or a different encoding to become "
                    "usable."
                ),
            )
        )
        steps.append(
            TransformationStep(
                operation="review_biased_importance",
                table=table,
                columns=(row["feature"],),
                parameters={
                    "impurity_importance": row["impurity_importance"],
                    "permutation_importance": row["permutation_importance"],
                    "noise_floor": value["noise_floor"],
                    "sentinel_impurity_ceiling": value["sentinel_impurity_ceiling"],
                },
                rationale=(
                    "Ranks near the top by an in-sample measure that favours "
                    "high-cardinality columns, and loses to noise out of sample. "
                    "Neither dropped nor trusted without a look: a decoy and a "
                    "masked real driver are indistinguishable here."
                ),
                evidence_ids=(item.id,),
                risk="medium",
            )
        )

    linear_score = value["linear_score"]
    advantage = value["tree_advantage"]
    model_score = value["model_score"]
    if (
        linear_score is not None
        and advantage is not None
        and advantage >= _TREE_GAIN_ABSOLUTE
        and model_score >= _TREE_GAIN_MIN_SCORE
        and (linear_score <= 0 or advantage / abs(linear_score) >= _TREE_GAIN_RELATIVE)
    ):
        drivers = [row["feature"] for row in rows if not row["below_noise_floor"]][:3]
        named = ", ".join(drivers) if drivers else "the top-ranked features"
        findings.append(
            Finding.create(
                title="A tree finds signal the linear probe missed",
                summary=(
                    f"A random forest reaches {metric} {model_score:.2f} on "
                    f"held-out rows where the {value['linear_model']} probe "
                    f"reaches {linear_score:.2f}, on the same split and the same "
                    "features. The relationship with the target is curved, "
                    "threshold-like, or driven by interactions."
                ),
                severity="medium",
                confidence=0.78,
                evidence_ids=(item.id,),
                recommendation=(
                    f"Before reaching for a linear model, engineer {named}: "
                    "buckets, splines, or explicit interaction terms. Or use a "
                    "model that fits curves natively."
                ),
                category=OBSERVATION,
            )
        )

    reliance = value["top_feature_solo_reliance"]
    solo_score = value["top_feature_solo_score"]
    if (
        rows
        and len(rows) >= _DOMINANCE_MIN_FEATURES
        and reliance is not None
        and reliance >= _DOMINANCE_SOLO_RATIO
        and model_score >= _DOMINANCE_MIN_SCORE
    ):
        top = rows[0]
        findings.append(
            Finding.create(
                title=f"{top['feature']} is almost the whole model",
                summary=(
                    f"Fitted on {top['feature']} alone, the same forest reaches "
                    f"{metric} {solo_score:.2f} on the held-out rows, against "
                    f"{model_score:.2f} for the full model with "
                    f"{value['feature_count']} features. The other "
                    f"{value['feature_count'] - 1} add almost nothing."
                ),
                severity="medium",
                confidence=0.74,
                evidence_ids=(item.id,),
                recommendation=(
                    f"Confirm {top['feature']} is known at prediction time and is "
                    "not recorded after the outcome. The leakage screen only "
                    "fires above 98% explained variance, so a column that is "
                    "merely a strong proxy for the target passes it and lands "
                    "here instead."
                ),
            )
        )
        steps.append(
            TransformationStep(
                operation="review_dominant_feature",
                table=table,
                columns=(top["feature"],),
                parameters={
                    "solo_score": solo_score,
                    "full_model_score": model_score,
                    "metric": value["metric"],
                    "target": target,
                },
                rationale=(
                    "A near-single-column model is either a genuinely strong "
                    "driver or a proxy recorded after the fact."
                ),
                evidence_ids=(item.id,),
                risk="high",
            )
        )

    # The drop list. Redundant partners are held back deliberately: permutation
    # splits their credit, so both score low while the information they share is
    # real, and dropping both would lose it.
    droppable = [
        row["feature"]
        for row in rows
        if row["below_noise_floor"]
        and not row["has_redundant_partner"]
        and not row["cardinality_biased"]
    ]
    held_back = [
        row["feature"]
        for row in rows
        if row["below_noise_floor"] and row["has_redundant_partner"]
    ]
    if droppable:
        rationale = (
            f"Each scored no better on held-out rows than a manufactured noise "
            f"column given to the same model, against a forest that reaches "
            f"{metric} {model_score:.2f}."
        )
        if held_back:
            rationale += (
                " "
                + ", ".join(held_back)
                + " also scored below the floor but are excluded: each has a "
                "near-interchangeable partner, and permutation splits credit "
                "between such a pair, so dropping both would lose real signal."
            )
        steps.append(
            TransformationStep(
                operation="drop_uninformative_features",
                table=table,
                columns=tuple(droppable),
                parameters={
                    "noise_floor": value["noise_floor"],
                    "metric": value["metric"],
                    "held_back_redundant": held_back,
                },
                rationale=rationale,
                evidence_ids=(item.id,),
                risk="low",
            )
        )
    return findings, steps
