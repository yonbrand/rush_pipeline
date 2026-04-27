"""
Leak-free, sklearn-compatible transformers used inside the CV Pipeline.

All transformers operate on numpy arrays (the CV harness drops column names).
Each is fit on the training fold only and applied to the test fold.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    SelectFromModel,
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.utils import resample


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class MissingRateFilter(BaseEstimator, TransformerMixin):
    """Drop columns whose missing rate exceeds *threshold*."""

    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xf = np.asarray(X, dtype=float)
        miss = np.isnan(Xf).mean(axis=0)
        self.keep_mask_ = miss < self.threshold
        if not self.keep_mask_.any():
            self.keep_mask_[0] = True
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_mask_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of each pair with |r| > threshold (keep the lower-index column)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if X.shape[1] <= 1:
            self.keep_idx_ = list(range(X.shape[1])) or [0]
            return self
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0.0)
        n = X.shape[1]
        drop = set()
        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j not in drop and corr[i, j] > self.threshold:
                    drop.add(j)
        self.keep_idx_ = sorted(set(range(n)) - drop)
        if not self.keep_idx_:
            self.keep_idx_ = [0]
        return self

    def transform(self, X):
        return X[:, self.keep_idx_]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _mi_classif(X, y):
    return mutual_info_classif(X, y, random_state=42)


def _mi_regression(X, y):
    return mutual_info_regression(X, y, random_state=42)


class ConsensusSelector(BaseEstimator, TransformerMixin):
    """Keep features voted by >= min_votes of {KBest, MI, L1}."""

    def __init__(self, task_type="classification", k=30, min_votes=2):
        self.task_type = task_type
        self.k = k
        self.min_votes = min_votes

    def fit(self, X, y):
        p = X.shape[1]
        k = min(self.k, p)

        score_func = f_classif if self.task_type == "classification" else f_regression
        skb_mask = SelectKBest(score_func, k=k).fit(X, y).get_support()

        mi_func = _mi_classif if self.task_type == "classification" else _mi_regression
        mi_scores = mi_func(X, y)
        mi_mask = np.zeros(p, dtype=bool)
        mi_mask[np.argsort(mi_scores)[-k:]] = True

        if self.task_type == "classification":
            l1_est = LogisticRegression(
                penalty="l1", solver="saga", C=0.1, max_iter=5000,
                class_weight="balanced", random_state=42)
        else:
            l1_est = Lasso(alpha=0.1, max_iter=5000, random_state=42)
        l1_mask = SelectFromModel(l1_est).fit(X, y).get_support()

        votes = skb_mask.astype(int) + mi_mask.astype(int) + l1_mask.astype(int)
        self.mask_ = votes >= self.min_votes
        if self.mask_.sum() < 5:
            self.mask_ = votes >= 1
        if self.mask_.sum() == 0:
            self.mask_ = np.ones(p, dtype=bool)
        return self

    def transform(self, X):
        return X[:, self.mask_]


class StabilitySelector(BaseEstimator, TransformerMixin):
    """Stability Selection (Meinshausen & Buehlmann 2010)."""

    def __init__(self, task_type="classification", n_bootstrap=50,
                 sample_fraction=0.7, threshold=0.6):
        self.task_type = task_type
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold

    def fit(self, X, y):
        n_samples, n_features = X.shape
        counts = np.zeros(n_features)
        sub_n = max(10, int(n_samples * self.sample_fraction))

        for i in range(self.n_bootstrap):
            idx = resample(np.arange(n_samples), n_samples=sub_n,
                           random_state=42 + i, replace=False)
            X_sub, y_sub = X[idx], y[idx]
            if self.task_type == "classification":
                est = LogisticRegression(
                    penalty="l1", solver="saga", C=0.1, max_iter=5000,
                    class_weight="balanced", random_state=42)
            else:
                est = Lasso(alpha=0.1, max_iter=5000, random_state=42)
            est.fit(X_sub, y_sub)
            coefs = np.abs(est.coef_).ravel()
            if len(coefs) == n_features:
                counts += (coefs > 1e-10).astype(int)

        scores = counts / self.n_bootstrap
        self.scores_ = scores
        self.mask_ = scores >= self.threshold
        if self.mask_.sum() < 5:
            top_k = min(20, n_features)
            self.mask_ = np.zeros(n_features, dtype=bool)
            self.mask_[np.argsort(scores)[-top_k:]] = True
        return self

    def transform(self, X):
        return X[:, self.mask_]


class MRMRSelector(BaseEstimator, TransformerMixin):
    """Min-Redundancy Max-Relevance (Peng, Long & Ding 2005)."""

    def __init__(self, task_type="classification", k=20):
        self.task_type = task_type
        self.k = k

    def fit(self, X, y):
        n_features = X.shape[1]
        k = min(self.k, n_features)
        if self.task_type == "classification":
            relevance = mutual_info_classif(X, y, random_state=42)
        else:
            relevance = mutual_info_regression(X, y, random_state=42)

        with np.errstate(invalid="ignore", divide="ignore"):
            corr_matrix = np.corrcoef(X, rowvar=False) ** 2
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        selected = []
        remaining = set(range(n_features))
        best = max(remaining, key=lambda i: relevance[i])
        selected.append(best); remaining.remove(best)
        for _ in range(k - 1):
            if not remaining:
                break
            sel_arr = np.array(selected)
            best_score = -np.inf
            best_feat = None
            for f in remaining:
                redundancy = corr_matrix[f, sel_arr].mean()
                score = relevance[f] - redundancy
                if score > best_score:
                    best_score = score
                    best_feat = f
            selected.append(best_feat); remaining.remove(best_feat)

        self.mask_ = np.zeros(n_features, dtype=bool)
        self.mask_[selected] = True
        return self

    def transform(self, X):
        return X[:, self.mask_]


class BlockPCATransformer(BaseEstimator, TransformerMixin):
    """Per-domain PCA (self-contained: imputes + scales internally)."""

    def __init__(self, feature_names, domain_map=None, variance_retained=0.80):
        self.feature_names = feature_names
        self.domain_map = domain_map or {}
        self.variance_retained = variance_retained

    def _assign_domains(self):
        assignments = {}
        assigned = set()
        for domain, prefixes in self.domain_map.items():
            idxs = []
            for i, name in enumerate(self.feature_names):
                if i not in assigned and any(name.startswith(p) for p in prefixes):
                    idxs.append(i); assigned.add(i)
            if idxs:
                assignments[domain] = idxs
        remaining = [i for i in range(len(self.feature_names)) if i not in assigned]
        if remaining:
            assignments["_passthrough"] = remaining
        return assignments

    def fit(self, X, y=None):
        self.domain_indices_ = self._assign_domains()
        self.imputers_ = {}; self.scalers_ = {}; self.pcas_ = {}
        for domain, idxs in self.domain_indices_.items():
            Xd = X[:, idxs].astype(float)
            imp = SimpleImputer(strategy="median"); Xd = imp.fit_transform(Xd)
            sc = StandardScaler(); Xd = sc.fit_transform(Xd)
            self.imputers_[domain] = imp; self.scalers_[domain] = sc
            if domain == "_passthrough" or Xd.shape[1] <= 2:
                self.pcas_[domain] = None
            else:
                pca = PCA(n_components=min(Xd.shape[0], Xd.shape[1]),
                          svd_solver="full")
                pca.fit(Xd)
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                k = int(np.searchsorted(cumvar, self.variance_retained)) + 1
                k = max(2, min(k, Xd.shape[1]))
                pca_final = PCA(n_components=k, svd_solver="full"); pca_final.fit(Xd)
                self.pcas_[domain] = pca_final
        return self

    def transform(self, X):
        blocks = []
        for domain, idxs in self.domain_indices_.items():
            Xd = X[:, idxs].astype(float)
            Xd = self.imputers_[domain].transform(Xd)
            Xd = self.scalers_[domain].transform(Xd)
            if self.pcas_[domain] is not None:
                Xd = self.pcas_[domain].transform(Xd)
            blocks.append(Xd)
        return np.hstack(blocks)
