import torch
import pandas as pd
import os

hidden_size = 128
output_size = 1
learning_rate = 0.01
epochs = 20
precision = torch.float64

seed = int(os.environ.get("RANDOM_STATE", 42))

data = pd.read_csv("data/train.csv", delimiter=',').drop(columns=['ID_code'])
data = (
    data.groupby('target', group_keys=False)
        .apply(lambda x: x.sample(n=10000, replace=True, random_state=42))
).sample(frac=1, random_state=42).reset_index(drop=True)

features = [feature for feature in data.columns if feature != 'target']
X = data[features]
y = data.target

X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))