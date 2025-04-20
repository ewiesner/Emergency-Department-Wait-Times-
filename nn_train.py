import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

train = pd.read_csv('train.csv')
train = train.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'race', 'pain', 'intime', 'outtime', 'chiefcomplaint'])
train['race_condensed'] = train['race_condensed'].fillna('Missing')

numeric_vars = ['admission_age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain_cleaned_advanced']
categorical_vars = ['arrival_transport', 'race_condensed']

numeric = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=100, random_state=2025)),
    ('scaler', StandardScaler())
])

impute_acuity = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=100, random_state=2025))
])

categorical = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric, numeric_vars),
        ('passthrough_cols', impute_acuity, ['acuity']),
        ('cat', categorical, categorical_vars)
    ])

X = train.drop(columns=['stay_length_minutes'])
y = train['stay_length_minutes'].values

preprocessor.fit(X)
X_processed = preprocessor.transform(X)

X_tensor = torch.tensor(X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed, dtype=torch.float32)
y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

train_ds = TensorDataset(X_tensor, y_tensor)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)


model = MLPRegressor(X_tensor.shape[1])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(50):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)  
    avg_loss = total_loss / len(train_dl.dataset)
    print(f"Epoch {epoch + 1}: RMSE Loss = {torch.sqrt(avg_loss):.4f}")


model.eval()
with torch.no_grad():
    y_pred_train = model(X_tensor).numpy().flatten()

print("Train R^2:", r2_score(y, y_pred_train))
print("Train RMSE:", mean_squared_error(y, y_pred_train, squared=False))

test = pd.read_csv('test.csv')
test = test.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'race', 'pain', 'intime', 'outtime', 'chiefcomplaint'])
test['race_condensed'] = test['race_condensed'].fillna('Missing')

X_test = test.drop(columns=['stay_length_minutes'])
y_test = test['stay_length_minutes'].values
X_test_processed = preprocessor.transform(X_test)
X_test_tensor = torch.tensor(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
                             dtype=torch.float32)

with torch.no_grad():
    y_pred_test = model(X_test_tensor).numpy().flatten()

print("Test R^2:", r2_score(y_test, y_pred_test))
print("Test RMSE:", mean_squared_error(y_test, y_pred_test, squared=False))
