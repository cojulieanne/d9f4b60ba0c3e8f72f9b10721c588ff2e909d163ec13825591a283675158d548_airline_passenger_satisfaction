import pandas as pd
import sys
from pathlib import Path
import re
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def get_data():
    """
    Loads the dataset from the specified CSV file located in the project's data directory.

    Returns:
        pd.DataFrame: The dataset loaded into a Pandas DataFrame.
    """
    data = pd.read_csv(project_root/'data'/'full_data.csv')
    return data

def check_uniqueness(df, column='id'):
    """
    Checks if the values in the specified column of the DataFrame are unique across all rows.

    Args:
        df (pd.DataFrame): The DataFrame to check.
        column (str, optional): The column name to check for uniqueness. Defaults to 'id'.

    Returns:
        bool: True if all values in the column are unique; raises AssertionError otherwise.

    Raises:
        AssertionError: If the number of unique values in the column does not match
                        the total number of rows in the DataFrame.

    Prints:
        A message indicating the status of the uniqueness check.
    """
    print(f"Checking uniqueness of column: '{column}'")

    unique_count = df[column].nunique()
    total_count = len(df)

    assert total_count == unique_count, (
        f"Uniqueness check failed: {column} has {unique_count} unique values "
        f"but DataFrame has {total_count} rows.\n"
    )

    print("All values are unique. Passed\n")
    return True

def split_data(df, target, threshold=0.2, stratify=True, random_state=42):
    """
    Splits the dataset and prints class distribution (event rates) in train and test.

    Args:
        df (pd.DataFrame): Full dataset.
        target (str): Target column name.
        threshold (float): Test set size (default 0.2).
        stratify (bool): Whether to stratify the split.
        random_state (int): Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_data = df[target] if stratify else None

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=threshold,
        stratify=stratify_data,
        random_state=random_state
    )

    print(f"Split complete: {len(X_train)} train rows, {len(X_test)} test rows")
    print("Event Rates (target=1):")
    print(f"Train: {y_train.mean():.4f}")
    print(f"Test : {y_test.mean():.4f}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


def impute_missing(train, test=None, columns=None, method='median'):
    """
    Imputes missing values in train and test using values derived from train only.

    Args:
        train (pd.DataFrame): Training DataFrame.
        test (pd.DataFrame, optional): Testing DataFrame.
        columns (list, optional): Columns to impute. Defaults to all numeric columns in train.
        method (str): Imputation method: 'mean', 'median', 'mode', 'min', 'max', or 'zero'.

    Returns:
        train_imputed, test_imputed (if test is provided)
    """

    train = train.copy()
    test = test.copy() if test is not None else None

    if columns is None:
        columns = train.select_dtypes(include='number').columns.tolist()

    for col in columns:
        print(col)
        if method == 'mean':
            value = train[col].mean()
        elif method == 'median':
            value = train[col].median()
        elif method == 'mode':
            value = train[col].mode()[0]
        elif method == 'min':
            value = train[col].min()
        elif method == 'max':
            value = train[col].max()
        elif method == 'zero':
            value = 0
        else:
            raise ValueError("Invalid method. Choose from: mean, median, mode, min, max, zero.")

        train_missing = train[col].isna().sum()
        train[col] = train[col].fillna(value)
        print(f"Imputed {train_missing} missing in data['{col}'] using {method} ({value:.2f})")

        if test is not None and col in test.columns:
            test_missing = test[col].isna().sum()
            test[col] = test[col].fillna(value)
            print(f"Imputed {test_missing} missing in test['{col}'] using train's {method} ({value:.2f})")

    return (train, test) if test is not None else train

def standardize_cols(data):
    """
    Standardizes DataFrame column names:
    - Lowercase
    - Replace punctuation and whitespace with underscores
    - Strip leading/trailing underscores

    Args:
        data (pd.DataFrame): The input DataFrame

    Returns:
        pd.DataFrame: DataFrame with cleaned column names
    """
    data = data.copy()
    cleaned_cols = []
    for col in data.columns:
        # Lowercase and replace punctuation/whitespace with _
        clean = re.sub(r'\W+', '_', col.lower())
        clean = clean.strip('_')
        cleaned_cols.append(clean)
    data.columns = cleaned_cols
    return data

def one_hot_encode(data, columns=None, drop_last=True):
    """
    Performs one-hot encoding on specified columns after lowercasing values.

    Args:
        data (pd.DataFrame): Input DataFrame.
        columns (list or None): Columns to encode (default: all object/category columns).
        drop_last (bool): Drop the last category to avoid multicollinearity.

    Returns:
        pd.DataFrame: One-hot encoded DataFrame with lowercase categorical values.
    """
    data = data.copy()

    if columns is None:
        columns = data.select_dtypes(include=['object', 'category']).columns.tolist()

    # Convert all categorical values to lowercase strings
    for col in columns:
        data[col] = data[col].astype(str).str.lower()
        data[col] = (
            data[col]
            .astype(str)
            .str.lower()
            .apply(lambda x: re.sub(r'\W+', '_', x))  # \W+ matches non-alphanumeric
        )

    # Apply one-hot encoding
    data = pd.get_dummies(data, columns=columns, drop_first=drop_last, dtype=int)

    print(f"One-hot encoded columns (lowercased): {columns}")
    return data


def label_encode(data, mappings):
    """
    Applies predefined label encodings to columns.

    Args:
        data (pd.DataFrame): The input DataFrame.
        mappings (dict): Dictionary of column-to-value mappings.

    Returns:
        pd.DataFrame: Encoded DataFrame.
    """
    data = data.copy()

    for col, mapping in mappings.items():
        data[col] = data[col].astype(str).str.lower().map(mapping)
        print(f"Applied mapping to '{col}': {mapping}")

    return data
