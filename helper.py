import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

def missing_values(df):
    missing_values_count = df.isnull().sum()
    total_values_count = df.shape[0]  # total rows
    missing_percentage = (missing_values_count / total_values_count) * 100

    result = pd.DataFrame({
        'total_count': total_values_count,
        'missing_count': missing_values_count,
        'missing_percentage': missing_percentage
    })

    return result



def prepare_bank_churn_data(csv_path='data/Bank_Churn.csv', random_state=0):
    """
    Loads and preprocesses the Bank Churn dataset.

    Steps:
    - Read CSV
    - Select features and target
    - Split into train/test
    - One-hot encode categoricals
    - Scale numerical features
    - Return transformed train/test sets and the fitted transformer
    """

    # read data
    df = pd.read_csv(csv_path)

    # select feature and target data
    X = df.drop(['CustomerId', 'Surname', 'Exited'], axis=1)
    y = df['Exited']

    # split train and test dataset
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, random_state=random_state
    )

    # encode categoricals and scale all others
    categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']

    encode_scale = ColumnTransformer(
        transformers=[
            ('ohe_categoricals', OneHotEncoder(drop='first'), categorical_cols)
        ],
        remainder=StandardScaler()
    )

    # package transformation logic
    transform = Pipeline([
        ('encode_scale', encode_scale)
    ])

    # apply transformations
    X_train = transform.fit_transform(X_train_raw)
    X_test = transform.transform(X_test_raw)

    return X_train, X_test, y_train, y_test
