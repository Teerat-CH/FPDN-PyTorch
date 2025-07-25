import pandas as pd
data = pd.read_csv("data/GPA.csv", delimiter=',').head(8000)
features = [feature for feature in data.columns if feature != 'gpa']
X = data[features]
y = data.gpa
print(X.shape)