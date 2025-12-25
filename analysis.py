from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics

from xgboost import XGBClassifier

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score, log_loss, precision_recall_curve, auc, average_precision_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 2000)
import matplotlib.pyplot as plt

# read bank churn data
df = pd.read_csv('data/Bank_Churn.csv')
print(df.head())

# select feature and target data
X = df.drop(['CustomerId','Surname','Exited'], axis=1)
y = df['Exited']


# split train and test dataset
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, random_state = 0)


# encode categoricals and scale all others
categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

encode_scale =  ColumnTransformer(
  transformers= [('ohe_categoricals', OneHotEncoder(categories='auto', drop='first'), categorical_cols)],
  remainder= StandardScaler()  # standardize all other features
  )

# package transformation logic
transform = Pipeline([
   ('encode_scale', encode_scale)
   ])

# apply transformations
X_train = transform.fit_transform(X_train_raw)
X_test = transform.transform(X_test_raw)

