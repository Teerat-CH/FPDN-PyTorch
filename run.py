import pandas as pd

data = pd.read_csv("data/train.csv", delimiter=',').drop(columns=['ID_code'])
data = (
    data.groupby('target', group_keys=False)
        .apply(lambda x: x.sample(n=10000, replace=True, random_state=42))
).sample(frac=1, random_state=42).reset_index(drop=True)
features = [feature for feature in data.columns if feature != 'target']
X = data[features]
y = data.target

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

print(X)
print(y)