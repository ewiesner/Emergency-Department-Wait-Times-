import pandas as pd
import numpy as np
import joblib as jl

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

train = pd.read_csv('train.csv')
train = train.drop(columns = ['subject_id', 'hadm_id', 'stay_id', 'race', 'pain', 'intime', 'outtime', 'chiefcomplaint'])

train['race_condensed'] = train['race_condensed'].fillna('Missing')

numeric_vars = ['admission_age', 'temperature', 'heartrate', 'resprate', 'o2sat', 
                'sbp', 'dbp', 'acuity', 'stay_length_minutes', 'pain_cleaned_advanced']
categorical_vars = ['gender', 'arrival_transport', 'race_condensed']


numeric = Pipeline(steps=[
    ('imputer', IterativeImputer(max_iter=100, random_state=2025)),
    ('scaler', StandardScaler())
])

categorical = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

impute_standardize = ColumnTransformer(
    transformers=[
        ('num', numeric, numeric_vars),
        ('cat', categorical, categorical_vars)
    ])

impute_standardize.fit(train)

train_imputed = impute_standardize.transform(train)

columns = numeric_vars +list(impute_standardize.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_vars))
imputed_df = pd.DataFrame(train_imputed, columns=columns)

jl.dump(impute_standardize, 'impute_standardize.pkl')
imputed_df.to_csv('train_imputed.csv', index=False)