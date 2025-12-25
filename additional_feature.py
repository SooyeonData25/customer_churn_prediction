import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 2000)

# read bank churn data
df = pd.read_csv('data/Bank_Churn.csv')


# select feature and target data
X_raw = df.drop(['CustomerId','Surname','Exited'], axis=1)
y = df['Exited']
print(X_raw.head())


# encode categoricals and scale all others
categorical_cols = ['Geography', 'Gender']

encode_scale =  ColumnTransformer(
  transformers= [('ohe_categoricals', OneHotEncoder(categories='auto', drop='first'), categorical_cols)],
  remainder= StandardScaler()  # standardize all other features
  )

# package transformation logic
transform = Pipeline([
   ('encode_scale', encode_scale)
   ])

# apply transformations
X = transform.fit_transform(X_raw)

# Train and score baseline model
# train the model
baseline_model = RandomForestClassifier(class_weight='balanced', random_state=0)
baseline_model.fit(X, y)

baseline_score = cross_val_score(
    baseline_model, X, y, cv=5, scoring="neg_mean_absolute_error"
)
baseline_score = -1 * baseline_score.mean()

print(f"MAE Baseline Score: {baseline_score:.4}")

# Create synthetic features
X_raw['BalanceSalaryRatio'] = X_raw['Balance']/(X_raw['EstimatedSalary']+1)
# X_raw['AgeTenureRatio'] = X_raw['Tenure']/(X_raw['Age'])


# apply transformations
X_extended = transform.fit_transform(X_raw)

# Train and score model with created ratio
# train the model
model = RandomForestClassifier(class_weight='balanced', random_state=0)
model.fit(X_extended, y)

score = cross_val_score(
    model, X_extended, y, cv=5, scoring="neg_mean_absolute_error"
)
score = -1 * score.mean()

print(f"MAE Score with Ratio Feature: {score:.4}")