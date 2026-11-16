import math
import random
from dataclasses import dataclass
from typing import List, Tuple


Vector = List[float]


def _sigmoid(x: float) -> float:
    if x < -50.0:
        return 0.0
    if x > 50.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class LogisticRegressionGD:
    """Simple logistic regression trained via gradient descent.

    This is a minimal binary classifier for demonstration only.
    """

    learning_rate: float = 0.1
    n_iter: int = 200
    l2: float = 0.0

    weights: Vector = None  # type: ignore
    bias: float = 0.0

    def fit(self, X: List[Vector], y: List[int]) -> None:
        if not X:
            self.weights = []
            self.bias = 0.0
            return

        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0 for _ in range(n_features)]
        self.bias = 0.0

        for _ in range(self.n_iter):
            grad_w = [0.0 for _ in range(n_features)]
            grad_b = 0.0
            for xi, yi in zip(X, y):
                z = self.bias + sum(w * xij for w, xij in zip(self.weights, xi))
                p = _sigmoid(z)
                diff = p - yi
                for j in range(n_features):
                    grad_w[j] += diff * xi[j]
                grad_b += diff

            # L2 regularization
            for j in range(n_features):
                grad_w[j] = grad_w[j] / n_samples + self.l2 * self.weights[j]
            grad_b = grad_b / n_samples

            for j in range(n_features):
                self.weights[j] -= self.learning_rate * grad_w[j]
            self.bias -= self.learning_rate * grad_b

    def predict_proba(self, X: List[Vector]) -> List[float]:
        if self.weights is None:
            return [0.5 for _ in X]
        probs: List[float] = []
        for xi in X:
            z = self.bias + sum(w * xij for w, xij in zip(self.weights, xi))
            probs.append(_sigmoid(z))
        return probs

    def predict(self, X: List[Vector]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]


@dataclass
class DecisionStump:
    feature_index: int
    threshold: float
    polarity: int  # 1 or -1
    vote: int      # 0 or 1

    def predict(self, X: List[Vector]) -> List[int]:
        preds: List[int] = []
        for xi in X:
            val = xi[self.feature_index] if 0 <= self.feature_index < len(xi) else 0.0
            cond = (val > self.threshold)
            if self.polarity == -1:
                cond = not cond
            preds.append(self.vote if cond else 1 - self.vote)
        return preds


def _find_best_stump(X: List[Vector], y: List[int], n_candidates: int = 10) -> DecisionStump:
    n_features = len(X[0]) if X else 0
    best_stump = DecisionStump(0, 0.0, 1, 1)
    best_err = 1.0
    if n_features == 0:
        return best_stump

    for _ in range(n_candidates):
        f = random.randint(0, n_features - 1)
        # pick random sample as threshold
        sample_idx = random.randint(0, len(X) - 1)
        thr = X[sample_idx][f]
        for polarity in (1, -1):
            for vote in (0, 1):
                stump = DecisionStump(f, thr, polarity, vote)
                preds = stump.predict(X)
                err = sum(1 for pi, yi in zip(preds, y) if pi != yi) / float(len(y))
                if err < best_err:
                    best_err = err
                    best_stump = stump
    return best_stump


@dataclass
class RandomForestLike:
    """A tiny random-forest–like ensemble using decision stumps.

    This is not a full-featured tree ensemble but suffices to provide a
    different inductive bias than logistic regression for the treatment.
    """

    n_estimators: int = 9
    n_candidates_per_stump: int = 20

    estimators: List[DecisionStump] = None  # type: ignore

    def fit(self, X: List[Vector], y: List[int]) -> None:
        self.estimators = []
        if not X:
            return
        for _ in range(self.n_estimators):
            stump = _find_best_stump(X, y, n_candidates=self.n_candidates_per_stump)
            self.estimators.append(stump)

    def predict_proba(self, X: List[Vector]) -> List[float]:
        if not self.estimators:
            return [0.5 for _ in X]
        probs: List[float] = []
        for xi in X:
            votes1 = 0
            for est in self.estimators:
                pred = est.predict([xi])[0]
                votes1 += pred
            p = votes1 / float(len(self.estimators))
            probs.append(p)
        return probs

    def predict(self, X: List[Vector]) -> List[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
