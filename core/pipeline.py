"""
Build the (preprocessing -> selection -> model) sklearn Pipeline plus the
inner-CV grid for each (selection x model) combination.

Default selection strategies are a curated subset of the original 10. The full
list is exposed as `ALL_SELECTION_STRATEGIES` for experiments that want them.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, SelectFromModel,
    f_classif, f_regression,
)
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA

from core.transformers import (
    MissingRateFilter, CorrelationFilter, ConsensusSelector,
    StabilitySelector, MRMRSelector, BlockPCATransformer,
    _mi_classif, _mi_regression,
)
from core.data import GAIT_DOMAINS

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# Default lean strategy set for the baseline (5 representatives that cover
# univariate, sparse, redundancy-aware, and unsupervised reduction).
DEFAULT_SELECTION_STRATEGIES = [
    "No Selection",
    "SelectKBest",
    "L1-based",
    "mRMR",
    "Stability",
]

ALL_SELECTION_STRATEGIES = [
    "No Selection", "SelectKBest", "Mutual Info", "L1-based", "Consensus",
    "PCA", "Stability", "mRMR", "Block PCA", "Block PCA + Stability",
]


def _preproc_steps(corr_threshold=0.95):
    return [
        ("miss_filter", MissingRateFilter(threshold=0.6)),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("corr_filter", CorrelationFilter(threshold=corr_threshold)),
        ("scale", StandardScaler()),
    ]


def build_pipeline_and_grid(model, model_params, selection_strategy, task_type,
                            feature_names=None, corr_threshold=0.95):
    """Return (Pipeline, combined_grid_dict). Nothing else."""
    extra_grid = {}

    if selection_strategy == "Block PCA":
        steps = [
            ("block_pca", BlockPCATransformer(
                feature_names=feature_names, domain_map=GAIT_DOMAINS,
                variance_retained=0.80)),
            ("model", model),
        ]
        extra_grid["block_pca__variance_retained"] = [0.80, 0.90]
        return Pipeline(steps), {**model_params, **extra_grid}

    if selection_strategy == "Block PCA + Stability":
        steps = [
            ("block_pca", BlockPCATransformer(
                feature_names=feature_names, domain_map=GAIT_DOMAINS,
                variance_retained=0.80)),
            ("rescale", StandardScaler()),
            ("select", StabilitySelector(task_type=task_type, n_bootstrap=30,
                                         sample_fraction=0.7, threshold=0.6)),
            ("model", model),
        ]
        return Pipeline(steps), {**model_params, **extra_grid}

    steps = _preproc_steps(corr_threshold=corr_threshold)

    if selection_strategy == "No Selection":
        pass
    elif selection_strategy == "SelectKBest":
        score_func = f_classif if task_type == "classification" else f_regression
        steps.append(("select", SelectKBest(score_func, k=20)))
        extra_grid["select__k"] = [10, 20, 40]
    elif selection_strategy == "Mutual Info":
        mi_func = _mi_classif if task_type == "classification" else _mi_regression
        steps.append(("select", SelectKBest(mi_func, k=20)))
        extra_grid["select__k"] = [10, 20, 30]
    elif selection_strategy == "L1-based":
        if task_type == "classification":
            est = LogisticRegression(penalty="l1", solver="saga", C=0.1,
                max_iter=5000, class_weight="balanced", random_state=42)
        else:
            est = Lasso(alpha=0.1, max_iter=5000, random_state=42)
        steps.append(("select", SelectFromModel(est)))
    elif selection_strategy == "Consensus":
        steps.append(("select", ConsensusSelector(task_type=task_type, k=30)))
    elif selection_strategy == "PCA":
        steps.append(("select", PCA(n_components=0.95, svd_solver="full")))
    elif selection_strategy == "Stability":
        steps.append(("select", StabilitySelector(
            task_type=task_type, n_bootstrap=30, sample_fraction=0.7, threshold=0.6)))
    elif selection_strategy == "mRMR":
        steps.append(("select", MRMRSelector(task_type=task_type, k=20)))
        extra_grid["select__k"] = [10, 20]
    else:
        raise ValueError(f"Unknown selection: {selection_strategy}")

    steps.append(("model", model))
    return Pipeline(steps), {**model_params, **extra_grid}


# ---------------------------------------------------------------------------
# Model registries
# ---------------------------------------------------------------------------

def get_clf_models():
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="lbfgs",
                class_weight="balanced", random_state=42),
            {"model__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [4, 8, None]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=-1,
                          tree_method="hist"),
            {"model__n_estimators": [200],
             "model__max_depth": [3, 5],
             "model__learning_rate": [0.05, 0.1],
             "model__reg_lambda": [1.0, 5.0]},
        )
    return models


def get_reg_models():
    models = {
        "ElasticNet": (
            ElasticNet(max_iter=5000, random_state=42),
            {"model__alpha": [0.01, 0.1, 0.5, 1.0],
             "model__l1_ratio": [0.3, 0.5, 0.7]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"model__n_estimators": [200], "model__max_depth": [4, 8, None]},
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist"),
            {"model__n_estimators": [200],
             "model__max_depth": [3, 5],
             "model__learning_rate": [0.05, 0.1],
             "model__reg_lambda": [1.0, 5.0]},
        )
    return models
