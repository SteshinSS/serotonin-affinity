import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(subsample=0.1)
train = np.load("data/preprocessed/train.npz")
val = np.load("data/preprocessed/val.npz")

gb.fit(train["X"], train["y"])
train_score = gb.score(train["X"], train["y"])
print(f"GB train score: {train_score}")

val_score = gb.score(val["X"], val["y"])
print(f"GB val score: {val_score}")
