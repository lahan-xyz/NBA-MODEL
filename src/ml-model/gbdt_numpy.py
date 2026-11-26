"""Minimal gradient-boosted regression trees in pure NumPy.
No sklearn, no compilation. Designed for small tabular datasets on Termux."""
import numpy as np


class _Tree:
    """One regression tree, fit to residuals via exact split search."""

    def __init__(self, max_depth=4, min_samples_leaf=20, max_features=None, rng=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = rng or np.random.default_rng(0)
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        n, p = X.shape
        pred = float(y.mean())
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return ('leaf', pred)

        feat_idx = np.arange(p) if self.max_features is None \
                   else self.rng.choice(p, self.max_features, replace=False)
        best_gain, best_feat, best_thr = 0.0, None, None
        parent_sse = float(((y - pred) ** 2).sum())

        for j in feat_idx:
            xj = X[:, j]
            order = np.argsort(xj)
            xs, ys = xj[order], y[order]
            # candidate split points: midpoints between distinct values
            # (skip if too few unique values)
            if len(np.unique(xs)) < 2:
                continue
            cut = np.where(np.diff(xs) > 0)[0]
            cut = cut[(cut + 1 >= self.min_samples_leaf) &
                      (n - cut - 1 >= self.min_samples_leaf)]
            if len(cut) == 0:
                continue
            # cumulative sum trick for fast SSE of every prefix
            cy = np.cumsum(ys)
            cy2 = np.cumsum(ys ** 2)
            k = cut + 1
            left_n, right_n = k.astype(float), (n - k).astype(float)
            left_sum, right_sum = cy[cut], cy[-1] - cy[cut]
            left_sq, right_sq = cy2[cut], cy2[-1] - cy2[cut]
            left_sse  = left_sq  - left_sum  ** 2 / left_n
            right_sse = right_sq - right_sum ** 2 / right_n
            gains = parent_sse - (left_sse + right_sse)
            k_best = int(np.argmax(gains))
            if gains[k_best] > best_gain:
                best_gain = float(gains[k_best])
                best_feat = j
                # threshold = midpoint between adjacent distinct values
                best_thr = float((xs[cut[k_best]] + xs[cut[k_best] + 1]) / 2)

        if best_feat is None:
            return ('leaf', pred)

        mask = X[:, best_feat] <= best_thr
        left = self._build(X[mask],  y[mask],  depth + 1)
        right = self._build(X[~mask], y[~mask], depth + 1)
        return ('split', best_feat, best_thr, left, right)

    def predict(self, X):
        out = np.empty(len(X))
        for i, row in enumerate(X):
            node = self.root
            while node[0] == 'split':
                _, feat, thr, left, right = node
                node = left if row[feat] <= thr else right
            out[i] = node[1]
        return out


class NumpyGBDT:
    """Gradient boosted regression trees with squared loss."""

    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=4,
                 min_samples_leaf=20, max_features=None, subsample=1.0,
                 random_state=0, verbose=False):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.subsample = subsample
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose
        self.trees = []
        self.base = 0.0
        self.feature_importances_ = None

    def fit(self, X, y, X_val=None, y_val=None, early_stopping_rounds=30):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, p = X.shape
        self.base = float(y.mean())
        pred = np.full(n, self.base)
        self.trees = []
        imp = np.zeros(p)
        best_val, best_iter = np.inf, 0
        best_trees = []

        for it in range(self.n_estimators):
            resid = y - pred
            if self.subsample < 1.0:
                idx = self.rng.choice(n, int(n * self.subsample), replace=False)
            else:
                idx = np.arange(n)
            tree = _Tree(self.max_depth, self.min_samples_leaf,
                         self.max_features, self.rng).fit(X[idx], resid[idx])
            # count feature usage for importance
            self._count_splits(tree.root, imp)
            self.trees.append(tree)
            pred += self.lr * tree.predict(X)

            if X_val is not None:
                vp = self.predict(X_val)
                val_mse = float(np.mean((vp - y_val) ** 2))
                if val_mse < best_val:
                    best_val = val_mse
                    best_iter = it + 1
                    best_trees = list(self.trees)
                elif it - best_iter >= early_stopping_rounds:
                    if self.verbose:
                        print(f"  early stop at iter {it} (best {best_iter})")
                    self.trees = best_trees
                    break
            if self.verbose and (it + 1) % 50 == 0:
                tr_mse = float(np.mean((pred - y) ** 2))
                msg = f"  iter {it+1:3d}  train_mse={tr_mse:.3f}"
                if X_val is not None:
                    msg += f"  val_mse={val_mse:.3f}"
                print(msg)

        self.feature_importances_ = imp / max(imp.sum(), 1e-9)
        return self

    def _count_splits(self, node, imp):
        if node[0] == 'leaf':
            return
        _, feat, _, left, right = node
        imp[feat] += 1
        self._count_splits(left, imp)
        self._count_splits(right, imp)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), self.base)
        for tree in self.trees:
            out += self.lr * tree.predict(X)
        return out
