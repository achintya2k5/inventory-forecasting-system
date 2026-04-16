from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def build_preprocessor(X):
    numeric_cols=X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols=X.select_dtypes(include=['object', 'bool']).columns

    numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
    ])

    categorical_pipeline=Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor=ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)   
    ])

    return preprocessor