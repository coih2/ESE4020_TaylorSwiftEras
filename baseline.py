"""Step 2: baseline evaluation. kNN on audio features to predict a song's album.

Reads songs_clean.csv from clean.py. The Taylor's Version file is not used here.
The test set is split off and left untouched; every number below comes from
cross-validation on the training set.

Run with:  pip install scikit-learn pandas numpy  &&  python clean.py  &&  python baseline.py
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 13
N_REPEATS = 10
K_GRID = list(range(1, 61))
FEATURES = ["danceability", "energy", "loudness", "acousticness", "instrumentalness",
            "valence", "tempo", "duration_ms", "explicit", "key", "mode"]

songs = pd.read_csv("songs_clean.csv")
X = songs[FEATURES].astype(float).to_numpy()
y = songs["album_name"].to_numpy()
groups = songs["group"].to_numpy()
print(f"N = {len(y)}, inputs = {X.shape[1]}, classes = {len(set(y))}")

# ---------- Split: grouped, stratified ~25% test set, untouched until the end ----------
splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
train_idx, test_idx = next(splitter.split(X, y, groups))
Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
assert not set(gtr) & set(groups[test_idx]), "song versions leaked across split"
print(f"Train: {len(train_idx)} songs; test (held out, not used): {len(test_idx)} songs")


def make_knn(k=5):
    # Scaling sits inside the pipeline, so it is refit on each training fold.
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))


def tuned_knn(seed):
    """kNN whose k is chosen by grouped, stratified 5-fold CV."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return GridSearchCV(make_knn(), {"kneighborsclassifier__n_neighbors": K_GRID},
                        cv=cv, scoring="accuracy")


# ---------- Tune k on the training set ----------
search = tuned_knn(SEED).fit(Xtr, ytr, groups=gtr)
best_k = search.best_params_["kneighborsclassifier__n_neighbors"]
print(f"\nGrid k = {K_GRID[0]}..{K_GRID[-1]}; CV chose k = {best_k} "
      f"(CV accuracy {search.best_score_:.3f}, optimistic because k was chosen on it)")
cv_curve = pd.DataFrame({"k": K_GRID, "mean": search.cv_results_["mean_test_score"],
                         "fold_sd": search.cv_results_["std_test_score"]})
mid = cv_curve[cv_curve.k.between(5, 50)]
print(f"CV accuracy for k = 5..50 ranges {mid['mean'].min():.3f} to {mid['mean'].max():.3f}; "
      f"typical fold-to-fold sd {mid['fold_sd'].median():.3f}")


# ---------- Error bars: repeated (nested) grouped 5-fold CV ----------
def repeated_cv(model_fn, nested):
    """Mean accuracy and SE over N_REPEATS shuffles of 5 grouped, stratified folds."""
    means, ses = [], []
    for r in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        model = model_fn(SEED + 100 + r)
        extra = {"params": {"groups": gtr}} if nested else {}
        s = cross_val_score(model, Xtr, ytr, groups=gtr, cv=outer, scoring="accuracy", **extra)
        means.append(s.mean())
        ses.append(s.std(ddof=1) / np.sqrt(len(s)))  # SE of one 5-fold estimate
    return np.mean(means), np.mean(ses)


knn_mean, knn_se = repeated_cv(tuned_knn, nested=True)        # k re-tuned in every fold
fixed_mean, fixed_se = repeated_cv(lambda s: make_knn(best_k), nested=False)
null_mean, null_se = repeated_cv(lambda s: DummyClassifier(strategy="most_frequent"), nested=False)

print(f"\nkNN, nested CV (headline):   {knn_mean:.3f} +/- {knn_se:.3f} SE  "
      f"(approx. 95%: {knn_mean - 2 * knn_se:.3f} to {knn_mean + 2 * knn_se:.3f})")
print(f"kNN, fixed k = {best_k}:          {fixed_mean:.3f} +/- {fixed_se:.3f} SE")
print(f"Null (majority album):        {null_mean:.3f} +/- {null_se:.3f} SE")
print(f"Random guessing:              {1 / len(set(y)):.3f}")
