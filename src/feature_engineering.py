from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(
    categorical_cols: list[str], 
    numerical_cols: list[str]
) -> Pipeline:
    """
    Construct a scikit-learn preprocessing pipeline for both numerical
    and categorical features.
    
    Args:
        categorical_cols: List of categorical feature names.
        numerical_cols: List of numerical feature names.
        
    Returns:
        Pipeline: A scikit-learn Pipeline object.
    """
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
        
    return Pipeline(steps=[('preprocessor', preprocessor)])
